import json

BASE_URL = "https://media.talkbank.org/childes/Eng-UK/MPI-EVA-Manchester/Gina/"

filenames = """
030001a.mp3
030002a.mp3
030003a.mp3
030004a.mp3
030005a.mp3
""".strip().split("\n")

urls = [f"{BASE_URL}{name.strip().strip("'")}" for name in filenames]
print(json.dumps(urls, indent=4))
