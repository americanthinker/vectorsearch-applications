from pytube import YouTube, Playlist
from camel_converter import to_snake
import transformers
from tqdm import tqdm
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class DataIngest:
    '''
    Arrangement of methods designed to download YouTube videos as audio files,
    transcribe the audio files, and join the metadata with the transcripts.
    '''
    def __init__(self, playlist_url: str, show_url: str=None) -> None:
        self.playlist_url = playlist_url
        self.primary_keys =  ['video_id',
                              'title',
                              'length_seconds',
                              'thumbnail_url',
                              'keywords',
                              'short_description',
                              'view_count',
                              'episode_num',
                              'episode_url',
                              'content'
                             ]
        self.show_url = show_url if show_url else Playlist(self.playlist_url).owner_url

    def get_playlist(self) -> Playlist:
        """Get playlist from youtube"""
        return Playlist(self.playlist_url)
    
    def download_audio(self, 
                       video_url: str,
                       index: int=None,
                       video_dir: str="videos/"
                       ) -> dict:
        """Download youtube video and return videoDetails"""
        
        yt = YouTube(video_url)
        filename = f"{os.path.join(video_dir, yt.video_id)}.mp4"
        (yt.streams
            .filter(only_audio=True, file_extension = "mp4")
            .order_by("abr")
            .desc()
            .first()
            .download(filename=filename)
        )
        details = yt.vid_info['videoDetails']
        if isinstance(index, int):
            details['episode_num'] = index
        details['episode_url']  = video_url
        #coerce string numbers to int for indexing on Weaviate
        if details.get('lengthSeconds'):
            details['lengthSeconds'] = int(details['lengthSeconds'])
        if details.get('viewCount'):
            details['viewCount'] = int(details['viewCount'])
        return self._convert_keys(details)

    def transcribe_audio(self,
                         pipeline: transformers.pipeline,
                         video_id: str,
                         return_timestamps: bool=False,
                         return_text: bool=False
                         ) -> None | str:
        """Transcribe downloaded video"""
        outputs = pipeline(f"videos/{video_id}.mp4",
                            chunk_length_s=30,
                            batch_size=24,
                            return_timestamps=return_timestamps,
                            )
        transcript = outputs['text'].strip().encode(encoding='utf-8').decode()
        with open(f"transcripts/{video_id}.txt", 'w') as f:
            f.write(transcript)
        if return_text:
            return transcript
    
    def _convert_keys(self, video_meta: dict) -> dict:
        """Convert keys to snake_case"""
        return {to_snake(k):v for k,v in video_meta.items()}
    
    def get_audio_files(self,
                        video_urls: list[str], 
                        video_dir: str='videos/',
                        return_dict: bool=True,
                        video_id_key: str='video_id'
                        ) -> dict | list[dict]:
        '''
        Sequential method for multiple video downloads. Downloads 
        YouTube videos as audio files and returns a dictionary with
        or a list of singular metadata dictionaries.
        '''
        meta_data = {} if return_dict else []
        for i, url in enumerate(tqdm(video_urls)):
            try:
                video_info = self.download_audio(url, i, video_dir)
                if return_dict:
                    meta_data[video_info[video_id_key]] = video_info
                else:
                    meta_data.append(video_info)
            except Exception as e:
                print(e)
                continue
        return meta_data
    
    def get_audio_files_threaded(self,
                                 video_urls: Playlist | list[str], 
                                 video_dir: str='videos/',
                                 return_dict: bool=True, 
                                 video_id_key: str='video_id'
                                 ) -> dict | list[dict] :
        '''
        Multithreaded concurrent method for multiple video downloads.
        Returns either a single dictionary with video_ids as keys or
        a list of singular metadata dictionaries.
        '''
        meta_data = {} if return_dict else []
        progress = tqdm("Downloading videos", total=len(video_urls))
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
            futures = [executor.submit(self.download_audio, url, i, video_dir) for i, url in enumerate(video_urls, start=1)]
            for future in as_completed(futures):
                try:
                    video_info = future.result()
                    if return_dict:
                        meta_data[video_info[video_id_key]] = video_info
                    else:
                        meta_data.append(video_info)
                except Exception as e:
                    print(f'Error due to: {e}')
                    continue
                progress.update(1)
        return meta_data
    
    def read_json(self, path: str) -> dict:
        """Read json file"""
        with open(path, 'r') as f:
            return json.loads(f.read()) 
    
    def write_json(self, data: dict, path: str) -> None:
        """Write json file"""
        with open(path, 'w') as f:
            f.write(json.dumps(data, indent=4)) 
    
    def get_transcript_paths(self, transcript_dir: str) -> list[Path]:
        """
        Get all transcript paths from directory.  Directory should contain
        only transcript files with the extension ".txt"
        """
        return [path for path in Path(transcript_dir).iterdir() if path.name.endswith('.txt')]
    
    def _get_thumbnail_url(self, episode_dict: str):
        """
        Get thumbnail url from episode dict.  If not found, returns self.show_url
        as a backup url.
        """
        thumbnail_dict = episode_dict.get('thumbnail')
        if thumbnail_dict:
            thumbnail_list = thumbnail_dict.get('thumbnails')
            if any(thumbnail_list):
                try:
                    return thumbnail_list[1]['url']
                except IndexError:
                    return thumbnail_list[0]['url']
            else: 
                print('thumbnail list is empty')
                return self.show_url
        else: 
            print('thumbnail key not found')
            return self.show_url

    def _create_temp_dict(self, meta_data: list[dict]) -> dict:
        """
        Create a temporary dictionary for joining metadata
        with transcripts
        """
        temp_dict = {}
        for d in meta_data:
            key = d['video_id']
            temp_dict[key] = d
        return temp_dict
    
    def _remove_keys(self, episode_dict: dict, keys_to_keep: list[str]=None) -> dict:
        """
        Remove uneeded keys from dictionary
        """
        keys_to_keep = keys_to_keep if keys_to_keep else self.primary_keys
        return {k:v for k,v in episode_dict.items() if k in keys_to_keep}
    
    def join_single_transcript_to_meta( self,
                                        transcript_path: Path, 
                                        temp_dict: dict, 
                                        content_key: str='content',
                                        thumbnail_key: str='thumbnail_url'
                                      ) -> None:
        """
        Executes in-memory insertion of content_key (transcript) and 
        thumbnail_url key into temp_dict (metadata dictionary).
        """
        video_id = transcript_path.name.split('.')[0]
        if video_id in temp_dict:
            with open(transcript_path) as f:
                text = f.read()
                temp_dict[video_id][content_key] = text
                temp_dict[video_id][thumbnail_key] = self._get_thumbnail_url(temp_dict[video_id])

    def join_all_transcripts_to_meta(self,
                                     transcript_paths: list[Path],
                                     meta_data: list[dict],
                                     content_key: str='content',
                                     thumbnail_key: str='thumbnail_url',
                                     keys_to_keep: list[str]=None
                                    ) -> dict:
        """
        Join all transcripts to metadata dictionary
        """
        temp_dict = self._create_temp_dict(meta_data)
        for path in transcript_paths:
            self.join_single_transcript_to_meta(path, temp_dict, content_key, thumbnail_key)
        temp_list = list(temp_dict.values())
        final_list = [self._remove_keys(d, keys_to_keep) for d in temp_list]
        return final_list