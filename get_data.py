import aiohttp
import asyncio
import os

# Base URL for the dataset
url = "https://physionet.org/files/challenge-2019/1.0.0/training/"
training_sets = ["training_setA", "training_setB"]


async def download_file(session, url, file_path):
    """
    Asynchronous file download function.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        async with session.get(url) as response:
            if response.status == 200:
                with open(file_path, "wb") as f:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        f.write(chunk)
                print(f"Downloaded {file_path}")
            else:
                print(f"Failed to download {file_path}: {response.status}")
    except Exception as e:
        print(f"Error downloading {file_path}: {str(e)}")

        await asyncio.sleep(1)  # Introduce a small delay for each request
        await download_file(session, url, file_path)


async def main():
    """
    Main function to orchestrate the download of files using aiohttp.
    """
    tasks = []
    async with aiohttp.ClientSession() as session:
        for i, training_set in enumerate(training_sets):
            # ids = range(1, 20644) if i == 0 else range(100001, 120001)
            ids = range(1, 5001) if i == 0 else range(100001, 105001)
            for id in ids:
                file_name = f"p{id:06d}.psv"
                file_url = f"{url}{training_set}/{file_name}?download"
                local_file_path = os.path.join("training", training_set, file_name)
                # Introduce a small delay for each request to avoid hitting the rate limit
                await asyncio.sleep(0.025)  # Adjust the sleep duration as needed
                task = asyncio.ensure_future(
                    download_file(session, file_url, local_file_path)
                )
                tasks.append(task)
        # Run all tasks concurrently
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
