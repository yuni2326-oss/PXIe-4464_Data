import numpy as np
import pytest
from pxie4464_daq.device.daq import MockDAQ

SAMPLE_RATE = 1024
CHUNK = 256
N_CHANNELS = 4


def test_mockdaq_configure_and_read():
    daq = MockDAQ()
    daq.configure(sample_rate=SAMPLE_RATE, record_length=CHUNK)
    daq.start()
    data = daq.read()
    assert data.shape == (N_CHANNELS, CHUNK)
    assert data.dtype == np.float64
    daq.stop()


def test_mockdaq_channels_have_different_frequencies():
    """각 채널이 서로 다른 주파수 사인파를 생성하는지 확인"""
    daq = MockDAQ()
    daq.configure(sample_rate=51200, record_length=51200)  # 1초치 데이터
    daq.start()
    data = daq.read()
    # FFT로 각 채널 주파수 추출
    freqs = np.fft.rfftfreq(51200, d=1/51200)
    for ch, expected_freq in enumerate([100, 200, 300, 400]):
        mag = np.abs(np.fft.rfft(data[ch]))
        dominant = freqs[np.argmax(mag[1:])+1]
        assert abs(dominant - expected_freq) < 2.0, f"CH{ch}: expected {expected_freq}Hz, got {dominant}Hz"
    daq.stop()


def test_mockdaq_context_manager():
    with MockDAQ() as daq:
        daq.configure(sample_rate=SAMPLE_RATE, record_length=CHUNK)
        daq.start()
        data = daq.read()
    assert data.shape == (N_CHANNELS, CHUNK)


def test_mockdaq_stop_before_start_does_not_raise():
    daq = MockDAQ()
    daq.configure(sample_rate=SAMPLE_RATE, record_length=CHUNK)
    daq.stop()  # stop without start should not raise


def test_mockdaq_read_before_start_raises():
    daq = MockDAQ()
    daq.configure(sample_rate=SAMPLE_RATE, record_length=CHUNK)
    with pytest.raises(RuntimeError):
        daq.read()


def test_mockdaq_configure_validates_inputs():
    daq = MockDAQ()
    with pytest.raises(ValueError):
        daq.configure(sample_rate=0, record_length=256)
    with pytest.raises(ValueError):
        daq.configure(sample_rate=1024, record_length=0)


hardware = pytest.mark.skipif(
    True, reason="requires PXIe-4464 hardware"
)


@hardware
def test_pxie4464_configure_and_read():
    from pxie4464_daq.device.daq import PXIe4464
    with PXIe4464(device_name="Dev1") as daq:
        daq.configure(sample_rate=51200, record_length=1024)
        daq.start()
        data = daq.read()
        assert data.shape == (4, 1024)
        assert data.dtype == np.float64
