import uvicorn


if __name__ == "__main__":
    
    args = dict(host="0.0.0.0", port=5000, reload=True, root_path="")
    uvicorn.run("src.app:app", **args)