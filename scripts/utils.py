
def save_results(results, file_path):
    with open(file_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
