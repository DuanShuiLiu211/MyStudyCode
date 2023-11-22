import hashlib
import os


def file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def remove_duplicates(folder_path):
    hashes = {}
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            filehash = file_hash(file_path)
            if filehash not in hashes:
                hashes[filehash] = file_path
            else:
                print(f"Removing duplicate file: {file_path}")
                print(f"Original file: {hashes[filehash]}")
                with open("./test.txt", "a") as fl:
                    fl.write(
                        f"Removing duplicate file: {file_path}\nOriginal file: {hashes[filehash]}\n"
                    )
                os.remove(file_path)


folder_path = (
    "/Users/WangHao/Sites/学习/Python/TuLiProject/TrainNetPipeline/datas/character"
)
remove_duplicates(folder_path)
