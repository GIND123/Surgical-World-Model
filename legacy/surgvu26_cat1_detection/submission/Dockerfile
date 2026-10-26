FROM --platform=linux/amd64 pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/opt/app/.cfg \
    YOLO_OFFLINE=True \
    MPLCONFIGDIR=/opt/app/.cfg \
    OMP_NUM_THREADS=4

RUN groupadd -r user && useradd -m --no-log-init -r -g user user \
 && mkdir -p /input /output /opt/app/.cfg && chown -R user:user /input /output /opt/app

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
# ultralytics pulls in opencv-python (needs libGL); swap it for the headless build
RUN python -m pip install --no-cache-dir --no-color -r /opt/app/requirements.txt \
 && python -m pip uninstall -y -q opencv-python opencv-contrib-python || true \
 && python -m pip install --no-cache-dir --no-color opencv-python-headless==4.10.0.84 \
 && python -c "import ultralytics, cv2, torch; print('deps ok', ultralytics.__version__, cv2.__version__, torch.__version__)" \
 && rm -rf /root/.cache

# `import ultralytics` in the layer above runs as root and creates
# $YOLO_CONFIG_DIR/settings.json owned by root, after the chown above. Harmless
# (ultralytics falls back to defaults) but it logs an error on every start.
RUN chown -R user:user /opt/app/.cfg

COPY --chown=user:user infer_core.py track_core.py inference.py /opt/app/
COPY --chown=user:user model /opt/app/model

USER user
ENTRYPOINT ["python", "inference.py"]
