import logging
from typing import Dict, Any

def save_results(results: Dict[str, Any], file_path: str) -> None:
    """
    Save the results dictionary to the specified file path as a plain text file.
    """
    try:
        with open(file_path, 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        logging.info("Results successfully saved to %s", file_path)
    except Exception as e:
        logging.error("Error saving results to %s: %s", file_path, e)
        raise