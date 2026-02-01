def load_file(filename: str):
  print(f"Reading file: {filename}")
  if not filename or len(filename) == 0:
    raise Exception("File name cannot be null/empty")
  with open(filename, 'r') as file:
    try:
      return file.read()
    except Exception as e:
      raise e