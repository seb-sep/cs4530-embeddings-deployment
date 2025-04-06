import modal

app = modal.App("example-get-started")

image = modal.Image.debian_slim().pip_install("fastapi[standard]")


@app.function(image=image)
@modal.fastapi_endpoint()
def square(x):
    print("This code is running on a remote worker!")
    x = int(x)
    return x**2


@app.local_entrypoint()
def main():
    print("the square is", square.remote(42))
