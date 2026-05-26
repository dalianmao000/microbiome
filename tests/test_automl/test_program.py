import pytest
from biomekit.automl.program import Program

class TestProgram:
    def test_default_program(self):
        program = Program()
        assert program.weights['performance'] == 0.5
        assert program.weights['stability'] == 0.25
        assert program.weights['bio_relevance'] == 0.25

    def test_custom_weights(self):
        program = Program(weights={
            'performance': 0.6,
            'stability': 0.3,
            'bio_relevance': 0.1,
        })
        assert program.weights['performance'] == 0.6

    def test_search_space_respected(self):
        program = Program()
        assert hasattr(program, 'search_space')
        assert program.search_space is not None

    def test_should_terminate_by_score(self):
        program = Program(termination_score=0.85)
        assert program.should_terminate(0.85, 1) == True
        assert program.should_terminate(0.86, 1) == True
        assert program.should_terminate(0.84, 1) == False

    def test_should_terminate_by_iterations(self):
        program = Program(max_iterations=10)
        assert program.should_terminate(0.5, 10) == True
        assert program.should_terminate(0.5, 9) == False

    def test_to_dict(self):
        program = Program()
        d = program.to_dict()
        assert 'weights' in d
        assert 'termination_score' in d
        assert 'max_iterations' in d