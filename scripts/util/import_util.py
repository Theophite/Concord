def script_imports(allow_zluda: bool = True):
    import logging
    import os
    import sys
    from pathlib import Path

    # Anti-fragmentation CUDA allocator config for Windows. The Linux launcher exports
    # PYTORCH_CUDA_ALLOC_CONF (lib.include.sh, gated on OT_CUDA_LOWMEM_MODE), but that bash include
    # never runs on Windows -> Windows used the DEFAULT allocator -> WDDM memory fragmentation and
    # silent demotion to shared memory (the slow-sampling / shared-spill wedge on tight 24 GB cards:
    # plenty of total free, but no contiguous block for the sampler's large allocations). Apply the
    # same config here, BEFORE torch is imported (the ZLUDA load below can pull it in). setdefault so
    # an explicit env value (or a future Windows launcher export) still wins; Linux is untouched.
    if sys.platform.startswith('win'):
        # PyTorch >=2.x renamed the var PYTORCH_CUDA_ALLOC_CONF -> PYTORCH_ALLOC_CONF (the old name
        # still works but warns). Set the NEW name (no warning on current PyTorch); also set the old
        # for pre-rename builds. setdefault so an explicit env value wins.
        _alloc_cfg = "garbage_collection_threshold:0.6,max_split_size_mb:128"
        os.environ.setdefault("PYTORCH_ALLOC_CONF", _alloc_cfg)
        if "PYTORCH_ALLOC_CONF" not in os.environ:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _alloc_cfg)

    # Filter out the Triton warning on startup.
    # xformers is not installed anymore, but might still exist for some installations.
    logging \
        .getLogger("xformers") \
        .addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

    # Insert ourselves as the highest-priority library path, so our modules are
    # always found without any risk of being shadowed by another import path.
    # 3 .parent calls to navigate from /scripts/util/import_util.py to the main directory
    onetrainer_lib_path = Path(__file__).absolute().parent.parent.parent
    sys.path.insert(0, str(onetrainer_lib_path))

    if allow_zluda and sys.platform.startswith('win'):
        from modules.zluda import ZLUDAInstaller

        zluda_path = ZLUDAInstaller.get_path()

        if os.path.exists(zluda_path):
            try:
                ZLUDAInstaller.load(zluda_path)
                print(f'Using ZLUDA in {zluda_path}')
            except Exception as e:
                print(f'Failed to load ZLUDA: {e}')

            from modules.zluda import ZLUDA

            ZLUDA.initialize()
