import neptune.new as neptune
run = neptune.init(
    project="NTLAB/test",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhNGRjNDgzOC04OTk5LTQ0YTktYjQ4Ny1hMTE4NzRjNjBiM2EifQ==",
)  # your credentials
run['train/test'].log(1)
run['train/test'].log(2)
run['train/test'].log(3)
run['train/test'].log(4)
run['train/test'].log(4)
run.stop()