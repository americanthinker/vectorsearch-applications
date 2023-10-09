######

# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.


import os, requests
from sentence_transformers import SentenceTransformer, InputExample, losses
import json, pickle
import argparse, random, time
from tqdm import tqdm
import torch, transformers, pickle, sklearn, numpy as np, pandas as pd, torch.nn as nn
from accelerate import Accelerator
from accelerate import notebook_launcher
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import TrainingArguments, Trainer, logging, AutoConfig
from rich.progress import track
from detoxify import Detoxify
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import yaml
import subprocess
from zipfile import ZipFile

warnings.filterwarnings('ignore')


class Synthetic_Query_Generation:

    def __init__(self,
                 corpus_file_path: str = None,
                 column_list: [str] = None,
                 sentences: [str] = None,
                 client=None,
                 index_name: str = None,
                 max_size: int = 9999, ) -> None:

        """
        Description:
        Initiate an object for synthetic query generation to take in data in three options:
        Option 1: read from local data folder in jsonl file
        Option 2: read from list of sentences
        Option 3: read from OpenSearch client by index_name

        Parameters:        
        corpus_file_path: str
            optional for option 1, file path for the jsonl file. 
        column_list: list
            optional for all options, input a list of columns to be used for query generation if None, default to choose all
            columns. When using multiple columns, the text from all of them will be concatenated
        sentences: list
            required for option 2, list of sentences to be used for query generation
        client: OpenSearch client
            required for option 3, the OpenSearch client to connect with OpenSearch Cluster
        index_name: str
            required for option 3, the index name to query the data
        max_size: int
            optional for option 3, the size limitation for querying data from OpenSearch. Default value to be 9999.

        Return:
            None
        """
        self.sentences = []
        self.input_ids = []
        self.attn_masks = []
        self.corpus = {}

        if corpus_file_path is None and sentences is None and client is None:
            raise Exception(
                "No data is provided. Please provide data in one of the three options: local jsonl file, list of sentences or OpenSearch client index_name")

        # Option 1: Read from jsonl file in local data folder 
        if corpus_file_path is not None:
            self.corpus_file = corpus_file_path

            if not len(self.corpus):
                print("Loading Corpus...")
                num_lines = sum(1 for _ in open(self.corpus_file, 'rb'))
                print(f"The number of lines in the json file are {num_lines}.")

                with open(self.corpus_file, 'r') as f:
                    for line in tqdm(f, total=num_lines):
                        line = json.loads(line)
                        idx = ''
                        column_count = 0
                        for column in line:
                            if column_count == 0:
                                idx = line.get(column)
                                self.corpus[idx] = {}
                            else:
                                self.corpus[idx][column] = line.get(column)

                            column_count += 1

                print("Loaded %d documents.", len(self.corpus))
                self.sentences = self.generate_sentences(column_list)

            else:
                raise Exception("Corpus file is empty! Please provide a non-empty file.")

        # Option 2: Read from list of sentences
        elif sentences is not None:
            self.sentences = sentences

        # Option 3: Read from OpenSearch client
        elif client is not None:
            response = client.search(index=index_name,
                                     body={"size": max_size,
                                           "query": {
                                               "match_all": {}}})
            corpus = response['hits']['hits']
            for records in corpus:
                idx = records['_id']
                self.corpus[idx] = {}
                source = records['_source']
                for column in source:
                    self.corpus[idx][column] = source.get(column)
            self.sentences = self.generate_sentences(column_list)

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl', bos_token='<|startoftext|>', eos_token='<|endoftext|>',
                                                  pad_token='<|pad|>', additional_special_tokens=["QRY:"])
        self.tokenizer = tokenizer

    def generate_synthetic_queries(self,
                                   tokenizer_max_length: int = 512,
                                   tokenize_data: bool = True,
                                   tokenized_corpus_path: str = None,
                                   compute_environment: str = None,
                                   num_machines: int = 1,
                                   num_gpu: int = 0,
                                   output_file_name: str = None,
                                   total_queries: int = 10,
                                   numseq: int = 10,
                                   toxic_cutoff: int = 0.1,
                                   tokens_to_word_ratio: float = 0.65
                                   ) -> pd.DataFrame:

        """
        Description:
        Generate synthetic queries given sentences

        Parameters:
        tokenizers_max_length: int
            set max length for tokenizers, default value as 512
        tokenize_data: bool
            flag to decide whether to tokenize data or not, defaults to True. Tokenizaton is a time-consuming process.
            If the corpus is tokenized once, this flag can be set to False. The script reads the tokenized corpus from
            the path provided in tokenized_corpus_path
        tokenized_corpus_path: str
            set path to store the tokenized corpus. If None then default to 'tokenized_corpus' folder in the current path
        overwrite: bool
            allow overwriting the same output filename # MS add comment about exception
        compute_environment: str
            optional, compute environment type to run model, if None, default using 'LOCAL_MACHINE'

        num_machines: int
            optional, number of machines to run model , if None, default using 1
        num_gpu: int
            optional, number of gpu to run model, default is 0
        output_file_name: str
            optional, output file name to save the generated queries,
            if None, default to 'synthetic_queries' in current directory
        total_queries: int
            Number of total queries to be generated
        numseq: int
            Number of queries that are generated by a model at a given time. Ideally total_queries = numseq, but this
            can lead to out of memory issues. So set numseq to an integer that is around 10 or less, and is a divisor of total_queries.
        toxic_cutoff: float
            Reject all queries that have a toxicity (as measured by the Detoxify model) score greater than toxic_cutoff.
        Return:
            None
        """

        batch_size = 1
        self.ratio = tokens_to_word_ratio
        # We set the batch size for the Synthetic query generator (SQG) model as 1. This is because the SQG model is a
        # 1.5B parameter model with a large GPU memory footprint. Larger batch sizes easily lead to out of memory issues.

        if tokenized_corpus_path is None:
            tokenized_corpus_path = os.path.join(os.getcwd(), 'tokenized_corpus')

        if tokenize_data == True:
            tokenized_corpus = self.generate_tokenized_corpus(tok_max_length=tokenizer_max_length)
            f = open(tokenized_corpus_path, "wb")
            pickle.dump(tokenized_corpus, f)
            f.close()
        else:
            f = open(tokenized_corpus_path, "rb")
            tokenized_corpus = pickle.load(f)
            f.close()

        if num_gpu > 1:
            self.set_up_accelerate_config(compute_environment=compute_environment,
                                          num_machines=num_machines,
                                          num_processes=num_gpu)

        self.generate(tokenized_corpus,
                      toxic_cutoff,
                      output_file_name=output_file_name,
                      num_processes=num_gpu,
                      batch_size=batch_size,
                      total_queries=total_queries,
                      numseq=numseq,
                      )

    def generate_tokenized_corpus(self, tok_max_length: int):
        """
        Description:
        Generate tokenized corpus from input data

        Parameters:
        tok_max_length: int
            set max length for tokenizers, default value as 512
        Return:
            tokenized corpus
        """

        print("Tokenizing corpus... This might take a while...")

        tokenized_corpus = GPT2Dataset_runtime_batch(self.sentences, self.tokenizer, self.ratio, tok_max_length)
        self.input_ids = tokenized_corpus.input_ids
        self.attn_masks = tokenized_corpus.attn_masks
        return tokenized_corpus

    def set_up_accelerate_config(self, compute_environment: str = None,
                                 num_machines: int = 1,
                                 num_processes: int = None) -> None:
        """
        Description:
        get default config setting based on the number of GPU on the machine
        if users require other configs, users can run !accelerate config for more options.

        Parameters:
        compute_environment: str
            optional, compute environment type to run model, if None, default using 'LOCAL_MACHINE'
        num_machines: int
            optional, number of machine to run model , if None, default using 1
        num_processes: int
            optional,  if None, default using the number of gpu available in the machine.
        Return:
            None
        """

        if compute_environment is None or compute_environment == 0:
            compute_environment = 'LOCAL_MACHINE'
        else:
            subprocess.run("accelerate config")
            return

        hf_cache_home = os.path.expanduser(
            os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
        )

        cache_dir = os.path.join(hf_cache_home, "accelerate")

        file_path = os.path.join(cache_dir + '/default_config.yaml')
        print('generated config file: at' + file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        default_file = [{'compute_environment': compute_environment,
                         'deepspeed_config':
                             {'gradient_accumulation_steps': 1,
                              'offload_optimizer_device': 'none',
                              'offload_param_device': 'none',
                              'zero3_init_flag': False,
                              'zero_stage': 2},
                         'distributed_type': 'DEEPSPEED',
                         'downcast_bf16': 'no',
                         'fsdp_config': {},
                         'machine_rank': 0,
                         'main_process_ip': None,
                         'main_process_port': None,
                         'main_training_function': 'main',
                         'mixed_precision': 'no',
                         'num_machines': num_machines,
                         'num_processes': num_processes,
                         'use_cpu': False}
                        ]

        with open(file_path, 'w') as file:
            yaml.dump(default_file, file)

    def generate(self,
                 tokenized_data,
                 toxic_cutoff,
                 output_file_name: str = None,
                 num_processes: int = 0,
                 batch_size: int = 1,
                 total_queries: int = None,
                 numseq: int = None,
                 ) -> None:
        """
        Description:
        Use Jupyter notebook launcher to run model with accelerate config to generate synthetic queries
        and detoxify them.

        Return:
            None
        """

        if output_file_name is None:
            output_file_name = 'synthetic_queries'

        if num_processes <= 1:
            use_accelerate = False
            self.run_model(tokenized_data, batch_size, total_queries, use_accelerate, numseq)  # start_method="spawn"
            self.detoxify(output_file_name, toxic_cutoff)
        else:
            use_accelerate = True
            if self.is_notebook():
                notebook_launcher(self.run_model, args=(tokenized_data, batch_size, total_queries, use_accelerate,
                                                        numseq),
                                  num_processes=num_processes)
                notebook_launcher(self.detoxify, args=(use_accelerate, toxic_cutoff), num_processes=num_processes)
            else:
                subprocess.run([
                                   "accelerate launch self.run_model(tokenized_data, batch_size, total_queries, use_accelerate, numseq)"])
                subprocess.run(["accelerate launch self.detoxify(use_accelerate, toxic_cutoff)"])

    def run_model(self,
                  tokenized_data,
                  batch_size,
                  total_queries,
                  use_accelerate,
                  numseq) -> None:
        """
        Description:
        Load the SQG model and generate queries

        Parameters:
        tokenized data: Object of GPT2Dataset class
        batch_size: int
        total_queries: int
            Number of total queries to be generated
        use_accelerate: bool
        numseq: int
            Number of queries that are generated by a model at a given time. Ideally total_queries = numseq, but this
            can lead to out of memory issues. So set numseq to an integer that is around 10 or less, and is a divisor of total_queries.
        Return:
            None
        """

        default_args = {
            "output_dir": "~/",
            "evaluation_strategy": "steps",
            "num_train_epochs": 1,
            "log_level": "error",
            "report_to": "none",
        }
        folder_name = "download_folder/"


        model_url = 'https://artifacts.opensearch.org/models/ml-models/amazon/gpt/GPT2_xl_sqg/1.0.0/GPT2_xl_sqg.zip'
        model_file_extension = '.zip'
        file_path = self.load_from_url(model_url, model_file_extension, folder_name)

        # loading config.json file from url
        model_config_url = 'https://artifacts.opensearch.org/models/ml-models/amazon/gpt/GPT2_xl_sqg/1.0.0/config.json'
        config_extension = '.json'
        _ = self.load_from_url(model_config_url, config_extension, folder_name)

        # unzip model_zip_file in the same folder
        unzip_path = os.path.join(os.getcwd(), folder_name)
        with ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(unzip_path)

        model = GPT2LMHeadModel.from_pretrained(unzip_path)
        soft = nn.Softmax(dim=0)
        tokenizer = self.tokenizer
        training_args = TrainingArguments(per_device_train_batch_size=batch_size, gradient_accumulation_steps=1,
                                          fp16=False,
                                          **default_args)

        train_dataloader = DataLoader(tokenized_data,
                                      batch_size=training_args.per_device_train_batch_size)

        learning_rate = 2e-5
        epsilon = 1e-8
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=epsilon)
        output = []

        if use_accelerate == True:
            accelerator = Accelerator()
            model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

            if accelerator.process_index == 0:
                print("The number of steps for creating queries that each process will take are ",
                      len(train_dataloader))
                print("Running on %s processes... " % accelerator.num_processes)

            with torch.no_grad():
                numgens = total_queries // numseq

                for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                    b_input_ids = batch[0]
                    for gen in range(numgens):
                        beam = model.module.generate(b_input_ids,
                                                     do_sample=True,
                                                     top_k=50,
                                                     max_length=len(b_input_ids[0]) + 25,
                                                     top_p=0.95,
                                                     temperature=1,
                                                     num_return_sequences=numseq,
                                                     repetition_penalty=1.2,
                                                     no_repeat_ngram_size=3,
                                                     return_dict_in_generate=True,
                                                     output_scores=True,
                                                     pad_token_id=tokenizer.eos_token_id)

                        for s in range(training_args.per_device_train_batch_size):
                            for i in range(numseq):
                                mydict = {}
                                counter = 0
                                probability = torch.tensor(1e3)
                                for k in range(len(b_input_ids[s]), beam["sequences"].shape[1]):
                                    if beam["sequences"][s * numseq + i][k] == 50256:
                                        break
                                    softmult = soft(beam["scores"][counter][s * numseq + i])[
                                        beam["sequences"][s * numseq + i][k]]
                                    probability = probability * softmult
                                    counter += 1
                                q = \
                                tokenizer.decode(beam["sequences"][s * numseq + i], skip_special_tokens=False).split(
                                    "QRY: ")[-1].split("<|")[0]
                                probability = probability.item()
                                passage = tokenizer.decode(b_input_ids[s])
                                mydict["probability"] = probability
                                mydict["query"] = q
                                mydict["passage"] = passage
                                output.append(mydict)

                if not os.path.exists(os.getcwd() + '/queries_before_detoxify'):
                    os.makedirs(os.getcwd() + '/queries_before_detoxify')

                file_path = str(
                    os.getcwd() + '/queries_before_detoxify/synthetic_query_' + str(accelerator.process_index))
                file = Path(file_path)
                f = open(file, "wb")
                pickle.dump(output, f)
                f.close()
        else:

            print("The number of steps for creating queries: ", len(train_dataloader))
            print("Running on CPU...")

            with torch.no_grad():
                numgens = total_queries // numseq
                for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                    b_input_ids = batch[0]
                    for gen in range(numgens):
                        beam = model.generate(b_input_ids,
                                              do_sample=True,
                                              top_k=50,
                                              max_length=len(b_input_ids[0]) + 25,
                                              top_p=0.95,
                                              temperature=1,
                                              num_return_sequences=numseq,
                                              repetition_penalty=1.2,
                                              no_repeat_ngram_size=3,
                                              return_dict_in_generate=True,
                                              output_scores=True,
                                              pad_token_id=tokenizer.eos_token_id)

                        for s in range(training_args.per_device_train_batch_size):
                            for i in range(numseq):
                                mydict = {}
                                counter = 0
                                probability = torch.tensor(1e3)
                                for k in range(len(b_input_ids[s]), beam["sequences"].shape[1]):
                                    if beam["sequences"][s * numseq + i][k] == 50256:
                                        break
                                    softmult = soft(beam["scores"][counter][s * numseq + i])[
                                        beam["sequences"][s * numseq + i][k]]
                                    probability = probability * softmult
                                    counter += 1
                                q = \
                                tokenizer.decode(beam["sequences"][s * numseq + i], skip_special_tokens=False).split(
                                    "QRY: ")[-1].split("<|")[0]

                                probability = probability.item()
                                passage = tokenizer.decode(b_input_ids[s])
                                mydict["probability"] = probability
                                mydict["query"] = q
                                mydict["passage"] = passage
                                output.append(mydict)

                if not os.path.exists(os.getcwd() + '/queries_before_detoxify'):
                    os.makedirs(os.getcwd() + '/queries_before_detoxify')

                file_path = str(os.getcwd() + '/queries_before_detoxify/synthetic_query')
                file = Path(file_path)
                f = open(file, "wb")
                pickle.dump(output, f)
                f.close()

        return None

    # Helper function
    def generate_sentences(self,
                           column_list: [str] = None) -> list:
        """
        Description:
        Create a list of input sentences to be used for synthetic query generation

        Return:
            A list of sentences
        """

        sentences = []

        for key in self.corpus:
            row = self.corpus[key]

            sent = ''
            for column in row:
                if column_list is None:
                    if row[column]:
                        sent = sent + row[column].strip()
                else:
                    if column in column_list:
                        sent = sent + row[column].strip()
            sentences.extend([sent])

        return sentences

    def detoxify(self, use_accelerate, toxic_cutoff, output_file_name: str = None):

        if output_file_name is None:
            output_file_name = 'clean_synthetic_queries'

        file_path = os.path.join(os.getcwd() + '/queries_before_detoxify/')
        file_list = []

        for root, dirnames, filenames in os.walk(file_path):
            for filename in filenames:
                file_list.append(os.path.join(root, filename))

        synthetic_query_data = []
        for files in file_list:
            f = open(files, "rb")
            synthetic_query_data.extend(pickle.load(f))
            f.close()

        df = pd.DataFrame(synthetic_query_data)
        df["id"] = df.index
        data = tasb_dataset_vanilla(df)

        model = Detoxify('unbiased')
        default_args = {"output_dir": os.getcwd()}
        training_args = TrainingArguments(per_device_train_batch_size=8, gradient_accumulation_steps=1, fp16=False,
                                          **default_args)
        train_dataloader = DataLoader(
            data,
            shuffle=False,
            batch_size=training_args.per_device_train_batch_size
        )

        optimizer = torch.optim.AdamW(model.model.parameters(), lr=2e-5)
        good_ids = []

        if use_accelerate == True:
            accelerator = Accelerator()
            model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

            if accelerator.process_index == 0:
                print("The total number of synthetic queries before detoxify is " + str(len(synthetic_query_data)))

            with torch.no_grad():
                for step, batch in enumerate(train_dataloader):
                    input_q = batch[0]
                    results = model.predict(input_q)
                    arr = []
                    for k, v in results.items():
                        arr.append(v)

                    targets = torch.tensor(arr)
                    targets = torch.sum(targets, axis=0)

                    ones = torch.ones(targets.shape)
                    zeros = torch.zeros(targets.shape)
                    targets = torch.where(targets > toxic_cutoff, ones, zeros)

                    if len(torch.where(targets == 0)[0]) > 0:
                        for i in torch.where(targets == 0)[0]:
                            good_ids.append(batch[2][i.item()].cpu().item())

            detoxified_df = df[df.id.isin(good_ids)]
            detoxified_df = detoxified_df.drop_duplicates()

            output = []
            for i in range(len(detoxified_df["id"])):
                mydict = {}
                mydict["probability"] = detoxified_df["probability"].values[i]
                mydict["query"] = detoxified_df["query"].values[i]
                mydict["passage"] = detoxified_df["passage"].values[i]
                output.append(mydict)

            if not os.path.exists(os.getcwd() + '/queries_after_detoxify'):
                os.makedirs(os.getcwd() + '/queries_after_detoxify')

            f = open(os.getcwd() + '/queries_after_detoxify/synthetic_queries_batch_' + str(accelerator.process_index),
                     "wb")
            pickle.dump(output, f)
            f.close()

            accelerator.wait_for_everyone()
            if accelerator.process_index == 0:
                file_list = []
                for root, dirnames, filenames in os.walk("queries_after_detoxify/"):
                    for filename in filenames:
                        if 'synthetic_queries_batch' in filename:
                            file_list.append(filename)
                with ZipFile(str(output_file_name + ".zip"), "w") as zipObj:
                    for file in file_list:
                        zipObj.write("queries_after_detoxify/" + file, file)
                print("Zip file is saved to" + os.getcwd() + "/" + output_file_name + ".zip")

        else:

            print("The total number of synthetic queries before detoxify is " + str(len(synthetic_query_data)))
            with torch.no_grad():
                for step, batch in enumerate(train_dataloader):
                    input_q = batch[0]
                    results = model.predict(input_q)
                    arr = []
                    for k, v in results.items():
                        arr.append(v)

                    targets = torch.tensor(arr)
                    targets = torch.sum(targets, axis=0)

                    ones = torch.ones(targets.shape)
                    zeros = torch.zeros(targets.shape)
                    targets = torch.where(targets > toxic_cutoff, ones, zeros)

                    if len(torch.where(targets == 0)[0]) > 0:
                        for i in torch.where(targets == 0)[0]:
                            good_ids.append(batch[2][i.item()].cpu().item())

            print(str(len(good_ids)) + ' good queries are kept after detoxify.')
            detoxified_df = df[df.id.isin(good_ids)]
            detoxified_df = detoxified_df.drop_duplicates()
            output = []
            for i in range(len(detoxified_df["id"])):
                mydict = {}
                mydict["probability"] = detoxified_df["probability"].values[i]
                mydict["query"] = detoxified_df["query"].values[i]
                mydict["passage"] = detoxified_df["passage"].values[i]
                output.append(mydict)

            if not os.path.exists(os.getcwd() + '/queries_after_detoxify'):
                os.makedirs(os.getcwd() + '/queries_after_detoxify')

            f = open(os.getcwd() + '/queries_after_detoxify/synthetic_queries_batch.p', "wb")
            pickle.dump(output, f)
            f.close()

            print("File is saved to " + os.getcwd() + "/queries_after_detoxify/synthetic_queries_batch_.p file.")

            file_list = []
            for root, dirnames, filenames in os.walk("queries_after_detoxify/"):
                for filename in filenames:
                    if 'synthetic_queries_batch' in filename:
                        file_list.append(filename)
            with ZipFile(str(output_file_name + ".zip"), "w") as zipObj:
                for file in file_list:
                    zipObj.write("queries_after_detoxify/" + file, file)
            print("Zip file is saved to" + os.getcwd() + "/" + output_file_name + ".zip")

        return None

    def is_notebook(self) -> bool:
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True  # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False
        except NameError:
            return False  # Probably standard Python interpreter

    def load_from_url(self, url, file_extension, folder_name):
        save_path = os.path.join(os.getcwd(), folder_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # If extension does not exist in end of url, append it
        if file_extension not in url.split("/")[-1]:
            filename = f'{url.split("/")[-1]}{file_extension}'
        # Else take the last part of the url as filename
        else:
            filename = url.split("/")[-1]

        if not os.path.isfile(str(save_path + filename)):
            r = requests.get(url)
            with open(str(folder_name + filename), 'wb') as f:
                f.write(r.content)
        return save_path + filename

    # @dataclass


class GPT2Dataset_runtime_batch(Dataset):
    def __init__(self, txt_list, tokenizer, ratio, max_input_length=512):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.max_len = max_input_length
        qry_id = self.tokenizer("QRY:")['input_ids'][0]

        count = 0
        print("Preparing input_ids and attention_mask... ")
        for txt in tqdm(txt_list, total=len(txt_list)):
            print(type(txt))
            print(txt)
            txt = " ".join(txt.split()[:int(self.max_len * ratio)])
            if type(txt) == str and len(txt) > 0:
                encodings_dict = self.tokenizer("<|startoftext|>" + txt + " QRY: ", truncation=True,
                                                max_length=max_input_length)
                if qry_id in encodings_dict['input_ids']:
                    self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                    self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
                else:
                    count += 1
        print(f'{count} number of documents out of {len(txt_list)} are discarded due to length constraints')

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

    def input_ids(self):
        return self.input_ids

    def attn_masks(self):
        return self.attn_masks

    def query(self):
        return self.query


class tasb_dataset_vanilla(Dataset):
    def __init__(self, df):
        self.queries = list(df["query"])
        self.passages = list(df["passage"])
        self.id = list(df["id"])

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.passages[idx], self.id[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Please enter argument to start semantic search training.')
    parser.add_argument("--corpus_file_path", type=str, default="corpus.jsonl")
    parser.add_argument("--column_list", type=list, default=None)
    parser.add_argument("--sentences", type=list, default=None)
    parser.add_argument("--client", default=None)
    parser.add_argument("--index_name", type=str, default=None)
    parser.add_argument("--max_size", type=int, default=9999)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--num_gpu", type=int, default=1)

    parser.add_argument("--tokenizer_max_length", type=bool, default=True)
    parser.add_argument("--tokenize_data", type=bool, default=True)
    parser.add_argument("--compute_environment", type=bool, default=True)
    parser.add_argument("--num_machines", type=int, default=1)
    parser.add_argument("--output_file_name", type=str, default=None)
    parser.add_argument("--total_queries", type=int, default=10)
    parser.add_argument("--numseq", type=int, default=10)
    parser.add_argument("--toxic_cutoff", type=float, default=0.1)

    args = parser.parse_args()
    corpus_file_path = args.corpus_file_path
    column_list = args.column_list
    sentences = args.sentences
    client = args.client
    index_name = args.index_name
    max_size = args.max_size
    output_path = args.output_path
    num_gpu = args.num_gpu

    tokenizer_max_length = args.tokenizer_max_length
    tokenized_data = args.tokenize_data
    compute_environment = args.compute_environment
    num_machines = args.num_machines
    output_file_name = args.output_file_name
    total_queries = args.total_queries
    numseq = args.numseq
    toxic_cutoff = args.toxic_cutoff

    print("Initiated synthetic query generation... ")

    training_object = Synthetic_Query_Generation(corpus_file_path, column_list, sentences, client, index_name,
                                                 max_size)

    training_object.generate_synthetic_queries(tokenizer_max_length, tokenize_data, tokenized_corpus_path,
                                               compute_environment, num_machines, num_gpu,
                                               output_file_name, total_queries, numseq, toxic_cutoff)


