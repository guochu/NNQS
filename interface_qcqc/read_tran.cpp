#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <map>
#include <vector>
#include <iostream>

typedef int8_t datatype;

extern "C" {
	void Hamtion_buffer(int n_qubits, int n_ele_max, int64_t * idxs,
			    datatype * pauli_mat12, int dim1_12, int dim2_12,
			    datatype * pauli_mat23, int dim1_23, int dim2_23,
			    double *coeffs, int idx_len, int coeff_len);

	void coupled_states(int64_t _state[], double eps,
			    int length, int64_t * _states, double *_coffes, int64_t *rcnt);

} 

typedef struct {
	double real;
	double imag;
} complex_t;

typedef struct {
	int **pauli_op;
	complex_t coffe;
	int pauli_num;
} tuple_t;

typedef struct {
	int n_qubits;
	int n_ele_max;
	int64_t *idxs;
	datatype *pauli_mat12;
	int dim1_12;
	int dim2_12;
	datatype *pauli_mat23;
	int dim1_23;
	int dim2_23;
	double *coeffs;
} MolecularHamiltonianIntOpt;

MolecularHamiltonianIntOpt *h =
    (MolecularHamiltonianIntOpt *) malloc(sizeof(MolecularHamiltonianIntOpt));

int global_n_qubits;
int global_n_qubit_op;

int state2id(const datatype * state, int offset, int row_len);

void Hamtion_buffer(int n_qubits, int n_ele_max, int64_t * idxs,
		    datatype * pauli_mat12, int dim1_12, int dim2_12,
		    datatype * pauli_mat23, int dim1_23, int dim2_23,
		    double *coeffs, int idx_len, int coeff_len);

void test_buffer();

void coupled_states(int64_t _state[], double eps,
		    int length, int64_t * _states, double *_coffes, int64_t *rcnt);

int state2id(const datatype * state, int offset, int row_len)
{
	int two = 1;
	int res = 0;
	for (int i = offset; i < offset + row_len; i++) {
		if (state[i] == 1) {
			res += two;
		}
		two *= 2;
	}
	return res;
}

//void test_buffer() {
//    printf("n_qubits is %d, n_ele_max is %d, dim1 is %d & %d, dim2 is %d & %d\n", h->n_qubits, h->n_ele_max, h->dim1_12, h->dim2_12, h->dim1_23, h->dim2_23);
//    printf("pauli12 here");
//    for (int i = 0; i < h->dim1_12 * h->dim2_12; ++i) {
//        printf("%d\t", h->pauli_mat12[i]);
//    }
//    printf("\n");
//
//    printf("pauli12 here");
//
//    for (int i = 0; i < h->dim1_23 * h->dim2_23; ++i) {
//        printf("%d\t", h->pauli_mat23[i]);
//    }
//    printf("\n");
//}

void Hamtion_buffer(int n_qubits, int n_ele_max, int64_t * idxs,
		    datatype * pauli_mat12, int dim1_12, int dim2_12,
		    datatype * pauli_mat23, int dim1_23, int dim2_23,
		    double *coeffs, int idx_len, int coeff_len)
{
	h->n_qubits = n_qubits;
	h->n_ele_max = n_ele_max;
	h->idxs = (int64_t *) malloc(idx_len * sizeof(int64_t));
	memcpy(h->idxs, idxs, idx_len * sizeof(int64_t));
	h->pauli_mat12 =
	    (datatype *) malloc(dim1_12 * dim2_12 * sizeof(datatype));

	memcpy(h->pauli_mat12, pauli_mat12, dim1_12 * dim2_12 *
	       sizeof(datatype));
	h->pauli_mat23 =
	    (datatype *) malloc(dim1_23 * dim2_23 * sizeof(datatype));

	memcpy(h->pauli_mat23, pauli_mat23,
	       dim1_23 * dim2_23 * sizeof(datatype));
	h->coeffs = (double *)malloc(coeff_len * sizeof(double));
	memcpy(h->coeffs, coeffs, coeff_len * sizeof(double));
	h->dim1_12 = dim1_12;
	h->dim2_12 = dim2_12;
	h->dim1_23 = dim1_23;
	h->dim2_23 = dim2_23;
}

void coupled_states(int64_t _state[], double eps, int length, int64_t * _states,
		    double *_coffes, int64_t *rcnt)
{
	int N = h->n_qubits;

	int target_value = -1;

	int _length_st = length;

//    printf("state input is\n");
//    for (int i = 0; i < length; ++i) {
//        printf("%lld\t", _state[i]);
//    }
//    printf("\n");

	int64_t state[_length_st];
	for (int i = 0; i < _length_st; ++i) {
		state[i] = (std::int64_t) 1;
		if (_state[i] == target_value) {
			state[i] = 0;
		}
	}

	int flag1 = N * h->n_ele_max;
	int flag2 = h->n_ele_max;

	int64_t *res_states = (int64_t *) malloc(flag1 * sizeof(int64_t));
	//printf("malloc 1 %lu\n", flag1 * sizeof(int64_t));
	double *res_coefs = (double *)malloc(flag2 * sizeof(double));
	//printf("malloc 2 %lu\n", flag2 * sizeof(double));

	for (int i = 0; i < flag1; ++i) {
		res_states[i] = 1;
	}

	memset(res_coefs, 0, flag2 * sizeof(double));

	int64_t res_cnt = 0;
	for (int sid = 0; sid < h->dim2_12; ++sid) {
		double coef = 0.0;
		int st = h->idxs[sid];
		int ed = h->idxs[sid + 1];

		for (int i = st; i < ed; ++i) {
			datatype x[h->dim1_23];
			for (int j = 0; j < h->dim1_23; ++j) {
				x[j] = h->pauli_mat23[i * h->dim1_23 + j];
			}
			datatype _tstate[h->dim1_23];
			int sum_1 = 0;

			for (int j = 0; j < h->dim1_23; ++j) {
				_tstate[j] = x[j] & state[j];
				if (_tstate[j] == 0) {
					sum_1++;
				}
			}
			int pow_1 = -1;
			coef += (pow(pow_1, sum_1)) * h->coeffs[i];
		}

		if (abs(coef) < eps) {
			continue;
		}
		res_cnt++;
		res_coefs[res_cnt - 1] = coef;

		int64_t pm12[h->dim1_12];
		for (int i = 0; i < h->dim1_12; ++i) {
			pm12[i] = h->pauli_mat12[sid * h->dim1_12 + i];
		}

		//printf("\n");
		int64_t nstate[h->dim1_12];
		for (int i = 0; i < h->dim1_12; ++i) {
			nstate[i] = state[i] ^ pm12[i];
			//printf("%d\t", nstate[i]);
		}
		//printf("\n");
		for (int i = 0; i < h->dim1_12; ++i) {
			if (nstate[i] == 0) {
				res_states[(res_cnt - 1) * N + i] =
				    target_value;
			}
		}
	}

    *rcnt = res_cnt;

    for (int i = N * res_cnt; i < flag1; ++i)
        res_states[i] = 0;


    memset(_states, 0, flag1 * sizeof(int));
	memset(_coffes, 0, flag2 * sizeof(double));
	memcpy(_states, res_states, flag1 * sizeof(int64_t));
	memcpy(_coffes, res_coefs, flag2 * sizeof(double));

//    printf("C state out\n");

//    for (int i = 0; i < flag1; ++i) {
//              //_states[i] = res_states[i];
//              printf("%lld\t", _states[i]);
//      }

//  printf("\n");
//    printf("C coffe out\n");

//    for (int i = 0; i < flag2; ++i) {
//        //_coefs[i] = res_coefs[i];
//              printf("%lf\t", _coffes[i]);
//      }

	fflush(stdout);

}

int imag_exp(int exp)
{
	int expm = exp % 4;
	switch (expm) {
	case 0:
		return 1;
	case 1:
		return 0;
	case 2:
		return -1;
	case 3:
		return 0;
	default:
		return 0;
	}
}

//tuple_t *read_qubit_op(const char *filename)
//{
//      // open the file
//      FILE *fp = fopen(filename, "rb");
//      if (fp == nullptr) {
//              fprintf(stderr, "Failed to open file: %s\n", filename);
//              return nullptr;
//      }
//      // read magic
//      double magic_number;
//      fread(&magic_number, sizeof(double), 1, fp);
//      if (magic_number != 11.2552) {
//              fprintf(stderr, "Invalid file format: %s\n", filename);
//              fclose(fp);
//              return nullptr;
//      }
//      // read qubit number
//      int n_qubits;
//      fread(&n_qubits, sizeof(int), 1, fp);
//      global_n_qubits = n_qubits;
//
//      tuple_t *qubit_op = nullptr;
//      int qubit_op_len = 0;
//
//      while (1) {
//              tuple_t qubit_tmp;
//              fread(&(qubit_tmp.coffe), sizeof(complex_t), 1, fp);
//              if (feof(fp)) {
//                      break;
//              }
//              int qubit_op_tmp[n_qubits];
//              fread(qubit_op_tmp, sizeof(int), n_qubits, fp);
//              int cnt = 0;
//              int *pos_tmp = nullptr;
//              int *op_tmp = nullptr;
//              for (size_t i = 0; i < n_qubits; i++) {
//                      if (qubit_op_tmp[i] != 0) {
//                              pos_tmp =
//                                  (int *)realloc(pos_tmp,
//                                                 ++cnt * sizeof(int));
//                              int ctmp = cnt - 1;
//                              pos_tmp[ctmp] = i;
//                              op_tmp =
//                                  (int *)realloc(op_tmp, cnt * sizeof(int));
//                              op_tmp[ctmp] = qubit_op_tmp[i];
//                      }
//              }
//
//              int **pauli_op_tmp = (int **)malloc(cnt * sizeof(int *));
//              for (int i = 0; i < cnt; i++) {
//                      pauli_op_tmp[i] = (int *)calloc(2, sizeof(int));
//              }
//              for (size_t i = 0; i < cnt; i++) {
//                      pauli_op_tmp[i][0] = pos_tmp[i];
//                      pauli_op_tmp[i][1] = op_tmp[i];
//                      // printf("%d\t%d\n",pauli_op_tmp[i][0], pauli_op_tmp[i][1]);
//              }
//
//              qubit_tmp.pauli_num = cnt;
//              qubit_tmp.pauli_op = pauli_op_tmp;
//              qubit_op_len++;
//              qubit_op =
//                  (tuple_t *) realloc(qubit_op,
//                                      qubit_op_len * sizeof(tuple_t));
//              // printf("%d\n", qubit_tmp.pauli_num);
//              qubit_op[qubit_op_len - 1] = qubit_tmp;
//              global_n_qubit_op = qubit_op_len;
//
//              // debuf info
//              // if (qubit_op_len != 1) {
//              // for (size_t i = 0; i < qubit_op[qubit_op_len - 1].pauli_num;
//              // i++) {
//              // //printf("%d\n", qubit_op[qubit_op_len - 1].pauli_num);
//              // if (i == qubit_op[qubit_op_len - 1].pauli_num - 1) {
//              // printf("(%d,%d)\n",
//              // qubit_op[qubit_op_len - 1].pauli_op[i][0],
//              // qubit_op[qubit_op_len - 1].pauli_op[i][1]);
//              // } else {
//              // printf("(%d,%d)\t",
//              // qubit_op[qubit_op_len - 1].pauli_op[i][0],
//              // qubit_op[qubit_op_len - 1].pauli_op[i][1]);
//              // }
//              // }
//              // }
//      }
//      fclose(fp);
//      return qubit_op;
//}

//MolecularHamiltonianIntOpt
//extract_indices_ham_int(tuple_t * qubit_op, double eps)
//{
//      int64_t N = global_n_qubits;
//      int64_t K = global_n_qubit_op;
//
//      datatype *pauli_mat12 = (datatype *) malloc(N * K * sizeof(datatype));
//      datatype *pauli_mat23 = (datatype *) malloc(N * K * sizeof(datatype));
//
//      memset(pauli_mat12, 0, N * K * sizeof(datatype));
//      memset(pauli_mat23, 0, N * K * sizeof(datatype));
//
//      std::map < int, std::vector < datatype > >pauli_mat12_dict;
//      std::map < int,
//          std::vector < std::vector < datatype > > >pauli_mat23_dict;
//      std::map < int, std::vector < double > >coeffs_dict;
//
//      int32_t row_idx = 0;
//
//      for (size_t i = 0; i < K; i++) {
//              int8_t cnt = 0;
//              if (abs(qubit_op[i].coffe.imag) > eps) {
//                      throw
//                          std::runtime_error
//                          ("Only support real-valued Hamiltonian!");
//              }
//              for (size_t j = 0; j < qubit_op[i].pauli_num; j++) {
//                      // printf("%d&%d&%d\t",
//                      // qubit_op[i].pauli_op[j][0],qubit_op[i].pauli_op[j][1],
//                      // qubit_op[i].pauli_num);
//                      if (qubit_op[i].pauli_op[j][1] == 1) {
//                              pauli_mat12[N * row_idx +
//                                          qubit_op[i].pauli_op[j][0]] = 1;
//                              // printf("%d\n %d\n", row_idx,
//                              // qubit_op[i].pauli_op[j][0]);
//                      } else if (qubit_op[i].pauli_op[j][1] == 2) {
//                              pauli_mat12[N * row_idx +
//                                          qubit_op[i].pauli_op[j][0]] = 1;
//                              pauli_mat23[row_idx * N +
//                                          qubit_op[i].pauli_op[j][0]] = 1;
//                              cnt++;
//                      } else {
//                              pauli_mat23[row_idx * N +
//                                          qubit_op[i].pauli_op[j][0]] = 1;
//                      }
//
//              }
//
//              /*
//               * fuse const calculation
//               */
//              double coeff = qubit_op[i].coffe.real * imag_exp(cnt);
//
//              int sid = state2id(pauli_mat12, row_idx * N, N);
//              std::vector < datatype > tmp12(N);
//              for (size_t i = 0; i < N; i++) {
//                      tmp12[i] = pauli_mat12[N * row_idx + i];
//              }
//              std::vector < datatype > tmp23(N);
//              for (size_t i = 0; i < N; i++) {
//                      tmp23[i] = pauli_mat23[N * row_idx + i];
//              }
//
//              if (coeffs_dict.count(sid)) {
//                      coeffs_dict[sid].push_back(coeff);
//                      pauli_mat23_dict[sid].push_back(tmp23);
//              } else {
//                      pauli_mat12_dict[sid] = tmp12;
//                      std::vector < std::vector < datatype > >_t_23;
//                      _t_23.push_back(tmp23);
//                      pauli_mat23_dict[sid] = _t_23;
//                      std::vector < double >_t_co;
//                      _t_co.push_back(coeff);
//                      coeffs_dict[sid] = _t_co;
//              }
//              row_idx += 1;
//      }
//
//      datatype pauli_mat12_buf[N * (pauli_mat12_dict.size())];
//      datatype pauli_mat23_buf[N * K];
//      double coeff_buf[K];
//      int64_t idxs[pauli_mat12_dict.size() + 1];
//
//      memset(pauli_mat12_buf, 0, N * (pauli_mat12_dict.size()));
//      memset(pauli_mat23_buf, 0, N * K);
//      memset(coeff_buf, 0, K);
//      memset(idxs, 0, (pauli_mat12_dict.size() + 1));
//
//      int num_idx = 0;
//      int cnt_mat12 = 0;
//      for (auto i = pauli_mat23_dict.begin(); i != pauli_mat23_dict.end();
//           i++) {
//              int num = i->second.size();
//              int sid_t = i->first;
//              for (size_t j = 0; j < num; j++) {
//                      for (size_t k = 0; k < i->second[j].size(); k++) {
//                              pauli_mat23_buf[(j + num_idx) * N + k] =
//                                  i->second[j][k];
//                      }
//
//              }
//
//              for (size_t i = 0; i < num; i++) {
//                      coeff_buf[num_idx + i] = coeffs_dict[sid_t][i];
//
//              }
//
//              for (size_t i = 0; i < pauli_mat12_dict[sid_t].size(); i++) {
//                      pauli_mat12_buf[cnt_mat12 * N + i] =
//                          pauli_mat12_dict[sid_t][i];
//              }
//
//              num_idx += num;
//              cnt_mat12++;
//              idxs[cnt_mat12] = num_idx;
//
//      }
//
//      MolecularHamiltonianIntOpt h;
//      memset(&h, 0, sizeof(h));
//      h.idxs = (int64_t *) malloc((cnt_mat12 + 1) * sizeof(int64_t));
//      h.pauli_mat12 =
//          (datatype *) malloc(N * (pauli_mat12_dict.size()) *
//                              sizeof(datatype));
//      h.pauli_mat23 = (datatype *) malloc(N * K * sizeof(datatype));
//      h.coeffs = (double *)malloc(K * sizeof(double));
//      h.n_qubits = N;
//      h.n_ele_max = pauli_mat12_dict.size();
//      h.dim1_12 = N;
//      h.dim2_12 = pauli_mat12_dict.size();
//      h.dim1_23 = N;
//      h.dim2_23 = K;
//
//      for (int i = 0; i < cnt_mat12 + 1; ++i) {
//              h.idxs[i] = idxs[i];
//      }
//
//      for (int i = 0; i < N * (pauli_mat12_dict.size()); ++i) {
//              h.pauli_mat12[i] = pauli_mat12_buf[i];
//      }
//
//      printf("\n");
//
//      for (int i = 0; i < N * K; ++i) {
//              h.pauli_mat23[i] = pauli_mat23_buf[i];
//      }
//
//      for (int i = 0; i < K; ++i) {
//              h.coeffs[i] = coeff_buf[i];
//      }
//
//      free(pauli_mat12);
//      free(pauli_mat23);
//
//      return h;
//
//}

int main()
{
//      tuple_t *qubit_op = NULL;
//      MolecularHamiltonianIntOpt h;
//      qubit_op =
//          read_qubit_op
//          ("/Users/zhoupengyu/Documents/ST_work/NNQS/interface_qcqc/mol_ham_data/c2/qubit_op.data");
//      printf("%d\n", global_n_qubit_op);
//      //h = extract_indices_ham_int(qubit_op, 0.0000001);
//      int64_t _state[4] = { 1, 1, -1, -1 };
//      int length = sizeof(_state)/sizeof(_state[0]);
//    int64_t *_states;
//    double *_coffes;
//      coupled_states(_state, 0.0000001, length, _states, _coffes);
//      // for (int i = 0; i < 8; ++i) {
//      // printf("%d\t", h.pauli_mat12[i]);
//      // }
//      //
//      // printf("\n");
//      //
//      // for (size_t i = 0; i < 4 * 15; i++) {
//      // if (((i + 1) % 4 == 0)) {
//      // printf("%d\n", h.pauli_mat23[i]);
//      // } else {
//      // printf("%d\t", h.pauli_mat23[i]);
//      // }
//      // }

	return 0;
}
