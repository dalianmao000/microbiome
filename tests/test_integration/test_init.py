import pytest

class TestModuleExports:
    def test_integration_module_imports(self):
        from biomekit.integration import MultiOmicsPipeline, MultiOmicsReport
        from biomekit.integration import (
            detect_input_format,
            align_omics_data,
            concat_with_labels,
            validate_omics_keys,
        )
        from biomekit.integration import (
            CCAAnalyzer,
            ProcrustesAnalyzer,
        )
        assert MultiOmicsPipeline is not None
        assert MultiOmicsReport is not None

    def test_main_import(self):
        from biomekit.integration import MultiOmicsPipeline
        assert callable(MultiOmicsPipeline)

    def test_all_exports_present(self):
        from biomekit import integration
        expected = [
            'MultiOmicsPipeline',
            'MultiOmicsReport',
            'CCAAnalyzer',
            'ProcrustesAnalyzer',
            'detect_input_format',
            'align_omics_data',
            'concat_with_labels',
            'validate_omics_keys',
        ]
        for name in expected:
            assert hasattr(integration, name), f"Missing export: {name}"