cd /Opus/torchtitan && \
pip install -r requirements.txt && \
# cd /Opus/src/opus-shim && \
# python setup.py develop && \
cd /Opus/scale-out/pypolatis-controller/PyPolatis-0.4.5 && \
python setup.py install



./opus-test/dp-2-tp-2-pp-2-eval/test-6-7-8-9-8gpu.sh