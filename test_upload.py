from tiktok_uploader.upload import upload_video
from tiktok_uploader.auth import AuthBackend

# Initialize auth with cookies
auth = AuthBackend(cookies='cookies/cookies.txt')

# Upload video
upload_video(
    'results/stories/A Truly Terrifying True Story_out.mp4',
    description='A truly terrifying true story #fyp #scary #storytime',
    auth=auth
) 