"""
Internal shared implementations for ftmwpipeline.

This module contains the core implementation functions that are shared between
the three interfaces: CLI commands, Pipeline class, and functional API.

The functions in this module are considered internal and should not be imported
directly by users. Instead, they should use one of the public interfaces:

- CLI: ftmwpipeline command-line interface
- Pipeline class: from ftmwpipeline import Pipeline
- Functional API: import ftmwpipeline as ftmw

Architecture:
- stage0_impl.py: Data import and FID visualization implementations
- stage1_impl.py: FT processing and spectrum visualization implementations  
- stage2_impl.py: Noise estimation and visualization implementations (future)
- shared_utils.py: Common utilities used across pipeline stages
"""