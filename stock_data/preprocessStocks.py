import json
import os


def extractData(fileName):
    with open(fileName, 'r') as file:
        data = json.load(file)

    time_series = data["Time Series (Daily)"]
    array = [[date, float(values["4. close"]), int(values["5. volume"])]
             for date, values in time_series.items()]

    symbol = data["Meta Data"]["2. Symbol"]

    # # Assign the array to the symbol name
    # globals()[symbol] = array

    return symbol, array


for file in os.listdir("../"):
    if file.endswith(".json"):
        file_path = os.path.join("../", file)
        try:
            symbol, array = extractData(file)
            output_file_name = f"{symbol}_DataProcessed.json"
            with open(output_file_name, "w") as f:
                json.dump(array, f)

            print(f"File created: {output_file_name}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
