BASE_URL = "https://media.talkbank.org/childes/Eng-UK/MPI-EVA-Manchester/Gina/"

def generate_download_url(filename):
    return f"{BASE_URL}{filename.strip()}"

filename = "030001a.mp3"
url = generate_download_url(filename)
print(url)
