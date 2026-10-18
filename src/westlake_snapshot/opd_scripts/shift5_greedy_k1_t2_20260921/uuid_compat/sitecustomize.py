"""Job-local vLLM UUID support, including model-inspection subprocesses."""
import os
if os.environ.get('OPD_VLLM_UUID_COMPAT') == '1':
    from vllm.platforms.interface import Platform
    from vllm.third_party import pynvml
    _original = Platform.device_id_to_physical_device_id.__func__
    def _uuid_to_nvml(cls, device_id):
        visible = os.environ.get(cls.device_control_env_var, '').split(',')
        if visible and visible[0].startswith('GPU-'):
            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByUUID(visible[device_id])
                return pynvml.nvmlDeviceGetIndex(handle)
            finally:
                pynvml.nvmlShutdown()
        return _original(cls, device_id)
    Platform.device_id_to_physical_device_id = classmethod(_uuid_to_nvml)
    Platform._opd_uuid_compat = True
