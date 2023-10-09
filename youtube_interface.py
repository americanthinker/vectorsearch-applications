from googleapiclient.discovery import build
from pytube import YouTube
from string import punctuation as punc
from pytube.exceptions import AgeRestrictedError, VideoPrivate, VideoUnavailable
from typing import List, Union
from loguru import logger
from dotenv import load_dotenv
load_dotenv('./.env', override=True)
import json
import re
import os

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

class YouTubeClient:
    """
    A class to interact with the YouTube API.
    """

    def __init__(self, 
                 api_service_name: str="youtube",
                 api_version: str="v3",
                 api_key = os.environ['YT_API_KEY']
                ) -> None:
        self.api_service_name = api_service_name
        self.api_version = api_version
        self.youtube = build(api_service_name, api_version, developerKey=api_key)

    def get_playlistItems(self, 
                          playlist_id: str, 
                          pageToken: str=None, 
                          max_results: int=1000
                          ) -> List[str]:
        """
        Makes a request to the YouTube API to get the playlist items for a given playlist id.
        """
        request = self.youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            pageToken=pageToken,
            maxResults=max_results)
        
        response = request.execute()
        return response

    def get_video_ids(self, playlist_id: str, max_results: int=1000) -> List[str]:
        """
        Returns a list of video ids for a given query.
        """
        response = self.get_playlistItems(playlist_id, max_results=max_results)

        video_ids = []
        items = response['items']
        for i in range(len(items)):
            video_ids.append(items[i]['contentDetails']['videoId'])
            
        next_page_token = response.get('nextPageToken')
        if next_page_token:
            more_pages = True
            page_count = 1
        
        while more_pages:
            page_count += 1
            page_response = self.get_playlistItems(playlist_id, next_page_token, max_results=50)
            
            for i in range(len(page_response['items'])):
                video_ids.append(page_response['items'][i]['contentDetails']['videoId'])
            logger.info(f'Total pages ripped: {page_count}')
            next_page_token = page_response.get('nextPageToken', False)
            if not next_page_token:
                more_pages = next_page_token
            
        return video_ids

    def write_video_ids_tofile(self, video_ids: List[str], out_path: str) -> None:
        """
        Writes a list of video ids as a text file.
        """
        with open(out_path, 'w') as f:
            for line in video_ids:
                f.write(line + '\n')
        logger.info('Video ids written to file.')


class PyTubeClient:
    """
    A class to interact with the PyTube API.
    """

    def __init__(self, playlist_id: str) -> None:
        self.playlist_id = playlist_id

    def create_video_url(self, video_id: str, playlist_id: str=None):
        """
        Creates a YouTube video url from a video id and playlist id. Video url 
        is a required param for creating a YouTube object.
        """
        if playlist_id is None:
            playlist_id = self.playlist_id
        return f'https://www.youtube.com/watch?v={video_id}&list={playlist_id}'
    
    def get_video_urls_from_ids(self, video_ids: List[str], playlist_id: str=None):
        """
        Creates a list of YouTube video urls from a list of video ids.
        """
        if playlist_id is None:
            playlist_id = self.playlist_id
        video_urls = [self.create_video_url(video_id, playlist_id) for video_id in video_ids]
        return video_urls
    
    def _wrangle_title(self, title: str, return_filename: bool=False) -> str:
        for p in punc:
            title = title.replace(p,' ')
        title = re.sub('\s+', ' ', title)
        title = re.sub('[_]+', '_', title)
        if return_filename:
            return title.strip().replace(' ', '_')
        else: return title

    def download_audio_file(self, 
                            video_url: str, 
                            data_folder: str, 
                            file_name: str=None,
                            skip_existing: bool=False,
                            return_metadata: bool=True
                            ) -> YouTube:
        """
        Downloads the audio from a YouTube video, given a video url. By default, 
        the audio file is saved to the data_folder prefixed with the video_id.
        Video ids are of the form: https://www.youtube.com/watch?v=VIDEO_ID
        """

        yt = YouTube(video_url)
        video_id = video_url.split('=')[1].split('&')[0]
        if file_name is None:
            try:
                title = yt.title
            except Exception as e:
                logger.info('Title not available. Using video id as filename.')
                logger.info(e)
                title = "NOT_AVAILABLE"
            file_name = self._wrangle_title(title, return_filename=True)
            file_name = f'{video_id}-{file_name}.mp3'
        try:
            audio = yt.streams.filter(only_audio=True).first()
            audio.download(output_path=data_folder, filename=file_name, skip_existing=skip_existing)
            logger.info(f'Audio downloaded and saved to {data_folder} as: {file_name}.')
        except (AgeRestrictedError, VideoPrivate, VideoUnavailable) as error:
            logger.info(f'Video {title}: {video_id} is being skipped due to {error}.')

        if return_metadata:
            return self.build_metadata_dict(yt) 
        else: return yt 

            
    def build_metadata_dict(self, ytobject: YouTube):
        """
        Extacts metadata from a YouTube object and converts it to a dictionary.
        """
        meta = {}
        meta['author'] = ytobject.author
        try:
            meta['title'] = ytobject.title
        except Exception as e:
            logger.info(e)
            meta['title'] = 'NOT_AVAILABLE'
        meta['video_id'] = ytobject.video_id
        meta['playlist_id'] = self.playlist_id
        meta['channel_id'] = ytobject.channel_id
        meta['description'] = ytobject.description if ytobject.description else "No description provided" 
        meta['keywords'] = ytobject.keywords
        try:
            meta['length'] = ytobject.length
        except Exception as e:
            logger.info(e)
            meta['length'] = -1
        meta['publish_date'] = ytobject.publish_date.strftime('%m-%d-%Y') if ytobject.publish_date else "No publish date provided"
        meta['thumbnail_url'] = ytobject.thumbnail_url
        try:
            meta['views'] = ytobject.views
        except Exception as e:
            logger.info(e)
            meta['views'] = -1
        meta['age_restricted'] = ytobject.age_restricted
        return meta
    
    def save_meta_toJSON(self, metadata: Union[dict, List[dict]], out_path: str) -> None:
        """
        Saves metadata to a JSON file.
        """
        if isinstance(metadata, dict):
            metadata = [metadata]
        with open(out_path, 'w') as f:
            json.dump(metadata, f)
        logger.info(f'Metadata saved as: {out_path}.')

if __name__ == "__main__":
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor, as_completed

    impact = PyTubeClient('PL8qcvQ7Byc3OJ02hbWJbHWePh4XEg3cvo')
    with open('./data/yt_video_ids.txt', 'r') as f:
        video_ids = f.read().splitlines()
    video_urls = impact.get_video_urls_from_ids(video_ids)
    metas = []
    count = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(impact.build_metadata_dict, YouTube(url)) for url in video_urls]
        # metas = [impact.build_metadata_dict(YouTube(url)) for url in tqdm(video_urls, "URLs")]
        for future in as_completed(futures):
            metas.append(future.result())
            count += 1
            if count % 25 == 0:
                logger.info(f'Created {count} metadata objects. {len(video_urls) - count} remaining.')
    # count = 0
    # for url in video_urls[235:]:
    #     meta = impact.download_audio_file(url, '/home/elastic/notebooks/vector-search-applications/data/podcast_mp3_files/ImpactTheory/')
    #     metas.append(meta)
    #     count += 1
    #     if count % 50 == 0:
    #         logger.info(f'Finished downloading {count} files. {len(video_urls) - count} files remaining.')
    logger.info(f'#of metas: {len(metas)}')
    impact.save_meta_toJSON(metas, './data/impact_theory_metadata.json')
    
