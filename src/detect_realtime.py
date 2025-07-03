def detect_realtime_frame(model, frame, conf=0.4):
    results = model(frame, stream=True, conf=conf)
    for r in results:
        return r.plot()
