BASE_URL = "https://media.talkbank.org/childes/Eng-UK/MPI-EVA-Manchester/Gina/"

filenames = """
030001a.mp3
030002a.mp3
030003a.mp3
030004a.mp3
030005a.mp3
""".strip().split("\n")

for file_name in filenames:
    url = f"{BASE_URL}{file_name.strip()}"
    print(f'"{url}"')
