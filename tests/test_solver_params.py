import pytest
import numpy as np
from quasar_solver.solver import Solver
from quasar_solver.qubo import QUBO

@pytest.fixture
def simple_qubo():
    return QUBO(np.array([[1, 0], [0, 1]]))

def test_invalid_initial_temp(simple_qubo):
    with pytest.raises(ValueError, match="initial_temp must be positive"):
        Solver(simple_qubo, initial_temp=-1.0)
    
    with pytest.raises(ValueError, match="initial_temp must be positive"):
         Solver(simple_qubo, initial_temp=0.0)

def test_invalid_final_temp(simple_qubo):
    with pytest.raises(ValueError, match="final_temp must be positive"):
        Solver(simple_qubo, initial_temp=10.0, final_temp=-0.1)
    
    with pytest.raises(ValueError, match="final_temp must be positive"):
        Solver(simple_qubo, initial_temp=10.0, final_temp=0.0)

def test_initial_less_than_final(simple_qubo):
    with pytest.raises(ValueError, match="initial_temp .* must be greater than final_temp"):
        Solver(simple_qubo, initial_temp=1.0, final_temp=10.0)

def test_invalid_cooling_rate(simple_qubo):
    with pytest.raises(ValueError, match="cooling_rate must be between 0 and 1"):
         Solver(simple_qubo, cooling_rate=1.1)
    
    with pytest.raises(ValueError, match="cooling_rate must be between 0 and 1"):
         Solver(simple_qubo, cooling_rate=-0.1)

    with pytest.raises(ValueError, match="cooling_rate must be between 0 and 1"):
         Solver(simple_qubo, cooling_rate=0.0)

    with pytest.raises(ValueError, match="cooling_rate must be between 0 and 1"):
         Solver(simple_qubo, cooling_rate=1.0)

def test_invalid_iterations(simple_qubo):
    with pytest.raises(ValueError, match="iterations_per_temp must be at least 1"):
        Solver(simple_qubo, iterations_per_temp=0)

def test_invalid_num_reads(simple_qubo):
    with pytest.raises(ValueError, match="num_reads must be at least 1"):
        Solver(simple_qubo, num_reads=0)

def test_valid_configuration(simple_qubo):
    # Should not raise
    solver = Solver(
        simple_qubo,
        initial_temp=10.0,
        final_temp=0.1,
        cooling_rate=0.99,
        iterations_per_temp=100,
        num_reads=1
    )
    assert solver.initial_temp == 10.0
    assert solver.final_temp == 0.1
    assert solver.cooling_rate == 0.99
    assert solver.iterations_per_temp == 100
    assert solver.num_reads == 1
