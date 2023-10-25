import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Parent link
parent_link = "https://docs.llamaindex.ai/en/stable/"

# Output directory
output_directory = "llamadocs_requests2"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Function to download a web page and its associated resources
def download_page(url: str, output_dir: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract and download all linked HTML pages
        for link in soup.find_all("a", href=True):
            href = link.get("href")
            if href.endswith(".html"):
                absolute_url = urljoin(url, href)
                _download_file(absolute_url, output_directory=output_dir)
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

# Function to download a file and save it locally
def _download_file(url: str, output_directory: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        file_name = os.path.join(output_directory, os.path.basename(url))
        with open(file_name, "wb") as file:
            file.write(response.content)
        print(f"Downloaded: {url}")

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

# Start the traversal from the parent link
download_page(parent_link)