from pytube import YouTube, Playlist
from camel_converter import to_snake
import transformers
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

class DataIngest:

    def __init__(self, playlist_url: str) -> None:
        self.playlist_url = playlist_url

    def get_playlist(self) -> Playlist:
        """Get playlist from youtube"""
        return Playlist(self.playlist_url)
    
    def download_audio(self, 
                       video_url: str,
                       index: int=None,
                       video_dir: str="videos/"
                       ) -> dict:
        """Download youtube video and return videoDetails"""
        
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
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
        return self._convert_keys(details)

    def transcribe_audio(self,
                         pipeline: transformers.pipeline,
                         video_meta: dict,
                         return_timestamps: bool=False,
                         return_text: bool=False
                         ) -> None | str:
        """Transcribe downloaded video"""
        video_id = list(video_meta.keys())[0]
        outputs = pipeline(f"videos/{video_id}.mp4",
                            chunk_length_s=30,
                            batch_size=24,
                            return_timestamps=return_timestamps,
                            )
        transcript = outputs['text'].strip().encode(encoding='utf-8').decode()
        with open(f"transcripts/{video_id}.txt", 'w') as f:
            f.write(transcript)
        video_meta[video_id]['text'] = transcript
        if return_text:
            return transcript
    
    def _convert_keys(self, video_meta: dict) -> dict:
        """Convert keys to snake_case"""
        return {to_snake(k):v for k,v in video_meta.items()}
    
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