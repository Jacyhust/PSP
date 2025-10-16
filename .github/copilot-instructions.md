# Copilot Instructions for PSP Codebase

## Overview
PSP (Proximity Graph with Spherical Pathway) is a C++ project implementing a novel approach to Maximum Inner Product Search (MIPS) using graph-based methods. The project is structured to support efficient indexing and searching for high-dimensional data, with a focus on scalability and performance.

## Codebase Structure
- **`src/`**: Core implementation files, including algorithms, indexing, and utility functions.
- **`include/`**: Header files defining class interfaces and utility functions.
- **`test/`**: Test cases for indexing and searching functionalities.
- **`datasets/`**: Contains datasets used for experiments.
- **`output/`**: Stores generated PSP indices and search results.
- **`scripts/`**: Scripts for running experiments.
- **`indexes/`**: Pre-built kNN graphs and PSP indices.

## Key Workflows

### Building the Project
1. Ensure prerequisites are installed:
   - GCC 4.9+ with OpenMP
   - CMake 2.8+
   - Boost 1.55+
   - (Optional) Faiss library
2. Compile the project:
   ```bash
   mkdir build && cd build
   cmake ..
   make -j
   ```

### Running the PSP Pipeline

#### Step 1: Build kNN Graph
Prepare a kNN graph using external libraries like Faiss.

#### Step 2: PSP Indexing
Run the following command to generate a PSP index:
```bash
./test/test_mips_index DATA_PATH KNNG_PATH L R Angle M PSP_PATH DIM
```
- Replace placeholders with appropriate paths and parameters.

#### Step 3: PSP Searching
Run the following command to perform a search:
```bash
./test/test_mips_search DATA_PATH QUERY_PATH PSP_PATH search_L K RESULT_PATH DIM
```
- Replace placeholders with appropriate paths and parameters.

## Project-Specific Conventions
- **Data Format**: Binary format (`.bin`) for datasets.
- **Indexing Parameters**:
  - `L`: Candidate pool size.
  - `R`: Maximum out-degree.
  - `Angle`: Minimum angle between edges.
  - `M`: Number of IP neighbors.
- **Search Parameters**:
  - `search_L`: Search pool size (must be larger than `K`).
  - `K`: Number of results.

## Testing
Test cases are located in the `test/` directory. Key test files include:
- `test_mips_index.cpp`: Tests for PSP indexing.
- `test_mips_search.cpp`: Tests for PSP searching.

Run tests after building the project to ensure correctness.

## Integration Points
- **External Libraries**:
  - Faiss: Used for building kNN graphs.
  - Boost: Required for various utilities.
- **Datasets**: Ensure datasets are in the correct format and placed in the `datasets/` directory.

## Performance Evaluation
- Metrics: Queries Per Second (QPS) and distance computations.
- Use the provided `evaluation.png` for reference.

## To-Do List
- Add Python wrapper for easier integration.
- Improve SIMD-related compatibility.
- Expand dataset support to include diverse norm distributions.

## Notes for AI Agents
- Focus on modularity: Most functionalities are encapsulated in `src/` and `include/`.
- Follow the parameter conventions for indexing and searching.
- Refer to the `README.md` for additional context on datasets and competitors.

For further questions, consult the `README.md` or the test cases in `test/`.