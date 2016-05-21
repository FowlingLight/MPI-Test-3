#include "mpi.h"
#include <iostream>
#include <fstream>

int SIZE;

void printMatrix(double *matrix) {
  for (int i = 0; i < (SIZE * SIZE); ++i) {
    if (i % SIZE == 0 && i != 0)
      std::cout << std::endl;
    std::cout << matrix[i] << "\t";
  }
  std::cout << std::endl;
}

double dotProduct(double *mat_a, double *mat_b, int row_a, int col_b) {
  double sum = 0;

  for (int i = 0; i < SIZE; i++)
    sum += mat_a[(row_a * SIZE) + i] * mat_b[(i * SIZE) + col_b];

  return sum;
}

double *multiplyStripe(double *stripe, double *mat_b, int stripe_size) {
  double *res = new double[stripe_size];

  for (int i = 0; i < stripe_size; i++)
    res[i] += dotProduct(stripe, mat_b, i / SIZE, i % SIZE);

  return res;
}

void generate_matrix() {
  double matrix[64] = {
    1, 2, 3, 4, 5, 6, 7, 8,
    2, 1, 2, 3, 4, 5, 6, 7,
    3, 2, 1, 2, 3, 4, 5, 6,
    4, 3, 2, 1, 2, 3, 4, 5,
    5, 4, 3, 2, 1, 2, 3, 4,
    6, 5, 4, 3, 2, 1, 2, 3,
    7, 6, 5, 4, 3, 2, 1, 2,
    8, 7, 6, 5, 4, 3, 2, 1
  };

  double matrix2[64] = {
    8, 7, 6, 5, 4, 3, 2, 1,
    7, 8, 7, 6, 5, 4, 3, 2,
    6, 7, 8, 7, 6, 5, 4, 3,
    5, 6, 7, 8, 7, 6, 5, 4,
    4, 5, 6, 7, 8, 7, 6, 5,
    3, 4, 5, 6, 7, 8, 7, 6,
    2, 3, 4, 5, 6, 7, 8, 7,
    1, 2, 3, 4, 5, 6, 7, 8
  };

  FILE *output = fopen("matA.dat", "wb");
  FILE *output2 = fopen("matB.dat", "wb");
  if (!output || !output2) {
    return;
  }

  fwrite(matrix, sizeof(double), 64, output);
  fwrite(matrix2, sizeof(double), 64, output2);

  fclose(output);
  fclose(output2);

}

int coordinator(int world_size, int argc, char **argv) {
  double *mat_a, *mat_b, *mat_c;

  mat_a = new double[SIZE * SIZE];
  mat_b = new double[SIZE * SIZE];
  mat_c = new double[SIZE * SIZE];

  FILE *input = fopen(argv[1], "rd");
  FILE *input2 = fopen(argv[2], "rd");
  if (!input || !input2) {
    return 0;
  }

  fread(mat_a, sizeof(double), SIZE * SIZE, input);
  fread(mat_b, sizeof(double), SIZE * SIZE, input2);

  fclose(input);
  fclose(input2);

  int stripe_size = (SIZE * SIZE) / world_size;
  double *partition = new double[stripe_size];

  MPI_Bcast(&stripe_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(mat_b, SIZE * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatter(mat_a, stripe_size, MPI_DOUBLE, partition, stripe_size,
	      MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double *res = multiplyStripe(partition, mat_b, stripe_size);

  MPI_Gather(res, stripe_size, MPI_DOUBLE, mat_c, stripe_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  printMatrix(mat_c);

  delete mat_a;
  delete mat_b;
  delete mat_c;
  delete partition;
  delete res;
}

int     participant(int world_rank) {

  int stripe_size;
  double *mat_b = new double[SIZE * SIZE];

  MPI_Bcast(&stripe_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  double *partition = new double[stripe_size];

  MPI_Bcast(mat_b, SIZE * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatter(NULL, stripe_size, MPI_DOUBLE, partition, stripe_size,
	      MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double *res = multiplyStripe(partition, mat_b, stripe_size);

  MPI_Gather(res, stripe_size, MPI_DOUBLE, NULL, stripe_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  delete partition;
  delete mat_b;
  delete res;
}

int     main(int argc, char **argv) {
  int world_size, world_rank;

  generate_matrix();

  if (argc != 4) {
    std::cerr << "Usage : mpirun -n 4 ./matrix_product matA.dat matB.dat rowSize" << std::endl;
    return -1;
  }

  SIZE = atoi(argv[3]);

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    coordinator(world_size, argc, argv);
  } else {
    participant(world_rank);
  }

  MPI_Finalize();

  return 0;
}
