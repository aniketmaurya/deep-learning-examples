from time import perf_counter


def benchmark_inference(model, inputs, trials=10):
    # warmup
    model(inputs)

    t0 = perf_counter()
    for i in range(trials):
        model(inputs)
    t1 = perf_counter()
    return t1 - t0

def benchmark_trainer(model, **kwargs):
    import lightning as L
    
    trainer = L.Trainer(
        max_epochs=2,
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    t0 = perf_counter()
    trainer.fit(model, **kwargs)
    t1 = perf_counter()
    return t1 - t0
