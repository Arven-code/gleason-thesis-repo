from pydub import AudioSegment

audio_file = "path/to/full_audio.wav"
start_ms = (1 * 60 + 7) * 1000
end_ms = (1 * 60 + 27) * 1000

audio = AudioSegment.from_file(audio_file)

snippet = audio[start_ms:end_ms]

output_path = "bobby_dialogue_01_07_to_01_27.wav"
snippet.export(output_path, format="wav")
print(f"✅ Extracted snippet saved as {output_path} (duration: {len(snippet)/1000:.2f} s)")
