# From https://dev.to/seijind/how-to-download-youtube-videos-in-python-44od
import pytube
import os
url = 'https://www.youtube.com/watch?v=RtUQ_pz5wlo'
youtube = pytube.YouTube(url, use_oauth=True, allow_oauth_cache=True)
video = youtube.streams.get_highest_resolution()
if not os.path.exists("./data"):
    os.mkdir("./data")
video.download("./data/")
