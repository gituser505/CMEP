#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <time.h>
#include <omp.h>

// --- xorshio64* random number generator --- //
typedef struct {
    uint64_t s[4];
} RngState;
uint64_t splitmix64( uint64_t *x ) ;
static inline uint64_t rotl(  uint64_t x, int k );
static inline uint64_t xoshio256plus( RngState *state );
static inline double rng( RngState *state );
void seed_rng( RngState *rng, int seed );
// --- xorshio64* random number generator --- //

typedef struct {
    int L;
    double J;
    double h;
    double beta;
    char *init;
    int N;
    int burnin;
    int decorr;
    uint32_t seed;
} Config;

typedef struct {
    int L;
    double J;
    double h;
    double beta;
    int8_t *lattice;
	int *n; 
    int *stack;
	double *magnet_data;
    double *energy_data;
    int N;
	int burnin;
    int decorr;
    RngState rng;
} Replica;

typedef struct {
    Config *config;
    Replica **replicas;
    int nthreads;
    double *betas;
    int nbetas;
    int sokal_c;
    char *path;
} Exp;

typedef struct {
    double val;
    double err;
} Obs;

typedef struct {
	double tau_M;
    double tau_E;
    Obs M;
	Obs E;
	Obs C;
	Obs chi;
	Obs bc;
} IsingObs;

typedef struct {
    double mean;
    double var;
    double bc;
} JackErr;

Config *get_config( int L, double J, double h, double beta, char *s, int N, int burnin, int decorr, unsigned seed );
Replica *init_replica( Config *config );
void free_replica( Replica *replica );
void init_neighbours( Replica *replica );
void init_lattice( Replica *replica, char *s );
Exp *init_experiment( Config *config, int nthreads, double *betas, int nbetas, int sokal_c, char *path);
void free_experiment( Exp *exp );

int wolff( Replica *restrict replica );
void wolff_sweep( Replica *restrict replica );
void run_simulation( Replica *restrict replica, char *path );
void run_experiment( Exp *exp );

IsingObs get_observables( Replica *replica );
static inline double magnetization( Replica *replica );
static inline double energy( Replica *replica );
static inline double mean( double *data, int n );
static inline double variance( double *data, int n );
static inline double binder( double *data, int n );
double *acf_fft( double *data, int n );
double tau_int( double *data, int n, int sokal_c );
JackErr jackknife_block( double *data, int n, int binsize );
JackErr jackknife( double *data, int n );
double *load_betas( char *fname, int *nbetas ); 
double clocking( void );


int main( void ) {

	int L = 32;
    int N = 3000;
	int burnin = 10*L*L;
	int decorr = 3;
	double J = 1;
	double h = 0;
	double beta = 0.44;

    int nbetas;
    double *betas = load_betas("betas.txt", &nbetas);
    int nthreads = omp_get_max_threads();
    int sokal_c = 5;

    int trial = 1;    
    char path[128];
    char mkdir_cmd[256];
    char *root = "/CMEP/data";
    while (1) {
        snprintf(path, sizeof(path), "%s/ising_%d_%d_%d/", root, L, N*nbetas, nbetas);
        if (access(path, F_OK) != 0) break;
    }
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", path);
    int err = system(mkdir_cmd);
    if ( err==0 ) printf("Created directory: %s\n", path);

    Config *config = get_config(L, J, h, beta, "random", N, burnin, decorr, (uint32_t)42);
    Exp *exp = init_experiment(config, nthreads, betas, nbetas, sokal_c, path);

    run_experiment(exp);
    free_experiment(exp);

/*      
    Config *config = get_config(L, J, h, beta, "random", N, burnin, decorr, (uint32_t)42);
    Replica *replica = init_replica(config);

	double t0 = clocking();
	run_simulation(replica, path);
	printf("CPU time: %lf\n", clocking()-t0);
	
	IsingObs o = get_observables(replica);
	printf("beta = %lf\n", beta);
	printf("tau_M = %lf\n", o.tau_M);
	printf("tau_E = %lf\n", o.tau_E);
	printf("M = %lf +- %lf\n", o.M.val, o.M.err);
	printf("E = %lf +- %lf\n", o.E.val, o.E.err);
	printf("C = %lf +- %lf\n", o.C.val, o.C.err);
	printf("chi = %lf +- %lf\n", o.chi.val, o.chi.err);
	printf("binder = %lf +- %lf\n", o.bc.val, o.bc.err);
	free(replica);
*/
    return 0;
}

Config *get_config( int L, double J, double h, double beta, char *s, int N, int burnin, int decorr, unsigned seed ) {
    Config *config = malloc( sizeof(Config) );
    config->L = L;
    config->J = J;
    config->h = h;
    config->beta = beta;
    config->init = s;
    config->N = N;
    config->burnin = burnin;
    config->decorr = decorr;
    config->seed = seed;
    return config;
}

Replica *init_replica( Config *config ) {
    Replica *replica = malloc( sizeof(Replica) );
    replica->N = config->N;
    replica->L = config->L;
    replica->J = config->J;
    replica->h = config->h;
	replica->beta = config->beta;
    replica->burnin = config->burnin;
    replica->decorr = config->decorr;
    replica->lattice = malloc( config->L*config->L * sizeof(int8_t) );
    replica->stack = malloc( config->L*config->L * sizeof(int) );
    replica->magnet_data = malloc( config->N * sizeof(double) );
    replica->energy_data = malloc( config->N * sizeof(double) );
    replica->n = malloc( 4*config->L*config->L * sizeof(int) );
    seed_rng(&replica->rng, config->seed);
    init_neighbours(replica);
    init_lattice(replica, config->init);
    return replica;
}

void init_neighbours( Replica *replica ) {
	int *n = replica->n;
	int L = replica->L;

	for ( int pos=0; pos<L*L; pos++ ) {
		int i = pos/L, j = pos%L;
		n[pos*4 + 0] = i*L + (j+1) % L;			// right neighbor
		n[pos*4 + 1] = i*L + (j-1+L) % L;		// left neighbor
		n[pos*4 + 2] = ((i+1) % L )*L + j;		// down neighbor
		n[pos*4 + 3] = ((i-1+L) % L )*L + j;    // up neighbor
    }
}

void init_lattice( Replica *replica, char *s ) {
    RngState *state = &replica->rng;
    int8_t *lattice = replica->lattice;
	int L = replica->L;

	if ( !strcmp(s, "up") ) 
		memset( lattice, 1, L*L*sizeof(int8_t) );
	else if ( !strcmp(s, "down") ) 
		memset( lattice, 0, L*L*sizeof(int8_t) );
	else if ( !strcmp(s, "random") )
        for (int i=0; i<L*L; i++)
            lattice[i] = ( rng(state) < 0.5 ) ? 1 : 0;
}

void free_replica( Replica *replica ) {
	free(replica->lattice);
    free(replica->stack); 
    free(replica->magnet_data);
    free(replica->energy_data);
    free(replica->n);
    free(replica); 
}

Exp *init_experiment( Config *config, int nthreads, double *betas, int nbetas, int sokal_c, char *path) { 
    Exp *exp = malloc( sizeof(Exp) );
    exp->config = config;
    exp->replicas = malloc( nthreads*sizeof(Replica*) );
    for ( int tid=0; tid<nthreads; tid++ )
        exp->replicas[tid] = init_replica(config);
    exp->nthreads = nthreads;
    exp->nbetas = nbetas;
    exp->betas = betas;
    exp->sokal_c = sokal_c;
    exp->path = path;
    return exp;
}

void free_experiment( Exp *exp ) {
    for ( int i=0; i<exp->nthreads; i++ )
        free_replica(exp->replicas[i]);
    free(exp->replicas);
    free(exp); 
}


int wolff( Replica *restrict replica ) {
    RngState *restrict state = &replica->rng;
    int8_t *restrict lattice = replica->lattice;
	int *restrict stack = replica->stack;
	int *restrict n = replica->n;
	double beta = replica->beta;
    double J = replica->J;
	int L = replica->L;

    double p_add = 1.0-exp(-2.0*J*beta);
	int stack_size = 0;
    int cluster_size = 1;
    int pos = (int)(L*L*rng(state));
	int8_t s = lattice[pos];
    lattice[pos] = 1-s;
    stack[stack_size++] = pos;
    do{ int pos = stack[--stack_size];
        for ( int k=0; k<4; k++ ) {
            int npos = n[pos*4+k];
            if ( lattice[npos] == s && rng(state) < p_add ) {
                lattice[npos] = 1-s;
                stack[stack_size++] = npos;
                cluster_size++;
			}
        }
    } while (stack_size > 0);
    return cluster_size;
}

void wolff_sweep( Replica *restrict replica ) {
    int L = replica->L;

    int lattices_flipped = 0;
    while (lattices_flipped < L*L)
        lattices_flipped += wolff(replica);
}

void run_simulation( Replica *restrict replica, char *path ) {
    double *restrict magnet_data = replica->magnet_data;
    double *restrict energy_data = replica->energy_data;
    int8_t *restrict lattice = replica->lattice;
    double beta = replica->beta;
	int decorr = replica->decorr;
	int burnin = replica->burnin;
	int N = replica->N;
    int L = replica->L;
    
    char fname_s[128], fname_b[128];
    snprintf(fname_s, sizeof(fname_s), "%s/%s", path, "lattice_samples.bin");
    snprintf(fname_b, sizeof(fname_b), "%s/%s", path, "beta_labels.bin");

    uint8_t *restrict lattice_buffer = malloc( N*L*L*sizeof(int8_t) );
    double *restrict beta_buffer = malloc( N*sizeof(double) );

    for (int i=0; i<burnin; i++)
        wolff_sweep(replica);
    for (int i = 0; i<N; i++) {
        for (int j=0; j<decorr; j++)
            wolff_sweep(replica);
        magnet_data[i] = magnetization(replica);
        energy_data[i] = energy(replica);
        beta_buffer[i] = beta;
        memcpy(lattice_buffer + (i*L*L), lattice, L*L);
    }
    #pragma omp critical(write_data_binary)
    {
        FILE *s = fopen(fname_s, "ab");
        FILE *b = fopen(fname_b, "ab");
        fwrite(lattice_buffer, sizeof(uint8_t), N*L*L, s);
        fwrite(beta_buffer, sizeof(double), N, b);
        fclose(s);
        fclose(b);
    }
    free(lattice_buffer);
    free(beta_buffer);
}

void run_experiment( Exp *exp ) {
    Replica **restrict replicas = exp->replicas;
    double *restrict betas = exp->betas;
    int nbeta = exp->nbetas;
    unsigned seed = exp->config->seed;
    char *path = exp->path;
    
    char fname[128];
    snprintf(fname, sizeof(fname), "%s/%s", path, "results.txt");
    FILE *f = fopen(fname, "w");
    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Replica *restrict local_rep = replicas[tid];
        RngState *state = &local_rep->rng;
       
        #pragma omp for schedule(dynamic)
        for ( int i=0; i<nbeta; i++) {
            seed_rng(state, seed + i);
            local_rep->beta = betas[i];
            init_lattice(local_rep, "random");
            run_simulation(local_rep, path);
            IsingObs o = get_observables(local_rep);
            double time = omp_get_wtime()-t0;
            
            #pragma omp critical(write_results_txt)
            {
                printf("CPU time: %lf %lf\n", betas[i], time);
                fprintf(f, "%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n", 
                        betas[i], o.tau_M, o.tau_E, o.M.val, o.M.err, o.E.val, o.E.err, 
                        o.chi.val, o.chi.err, o.C.val, o.C.err, o.bc.val, o.bc.err);
                fflush(f);
            }
        }
    }
    fclose(f);
}


IsingObs get_observables( Replica *replica ) {
	double *restrict magnet_data = replica->magnet_data;
	double *restrict energy_data = replica->energy_data;
	int N = replica->N;

    JackErr M_errs = jackknife_block(magnet_data, N, 1);
    JackErr E_errs = jackknife_block(energy_data, N, 1);

    IsingObs o;
    o.tau_M = tau_int(magnet_data, N, 5);
	o.tau_E = tau_int(energy_data, N, 5);
    o.M.val = mean(magnet_data, N);
    o.E.val = mean(energy_data, N);
    o.chi.val = variance(magnet_data, N);
    o.C.val = variance(energy_data, N);
    o.bc.val = binder(magnet_data, N);
    o.M.err = M_errs.mean;
    o.E.err = E_errs.mean;
    o.chi.err = M_errs.var;
    o.C.err = M_errs.var;
    o.bc.err = M_errs.bc;
	return o;
}

static inline double magnetization( Replica *replica ) {
    int8_t *restrict lattice = replica->lattice;
	int L = replica->L;
    
    int M = 0;
	#pragma omp simd reduction(+:M)
	for ( int p=0; p<L*L; p++ )
		M += lattice[p];
	return fabs(2.0f*M/(L*L) - 1.0f);
}

static inline double energy( Replica *replica ) {
    int8_t *restrict lattice = replica->lattice;
    int *restrict n = replica->n;
	int L = replica->L;
	double J = replica->J;

	int E = 0;
    #pragma omp simd reduction(+:E)
    for ( int p=0; p<L*L; p++ ) {
        int s = 2*lattice[p]-1;
        int s1 = 2 * lattice[n[p*4 + 1]] - 1;
        int s2 = 2 * lattice[n[p*4 + 2]] - 1;
        E -= (s1 + s2) * s;
    }
    return J*E / (L*L);
}

static inline double mean( double *data, int n ) {
	double avg = 0.0;
	#pragma omp simd reduction(+:avg)
	for ( int i=0; i<n; i++) 
		avg += data[i];
	return avg/n;
}

static inline double variance( double *data, int n) {
	double m = mean(data,n), var = 0.0;
	#pragma omp simd reduction(+:var)
	for ( int i=0; i<n; i++) {
		double d = data[i]-m;
		var += d*d;
	}
	return var/(n-1.0);
}

static inline double binder( double *data, int n ) {
	double m2 = 0.0, m4 = 0.0;
	#pragma omp simd reduction(+:m2,m4)
	for ( int i=0; i<n; i++ ) {
		double d = data[i];
		double d2 = d*d;
		m2 += d2;
		m4 += d2*d2;
	}
	return 1.0 - n*m4/(3.0*m2*m2);
}

double *acf_fft( double *data, int n ) {
	int pad = 2*n;
    double *in = fftw_malloc( pad*sizeof(double) );
    fftw_complex *out = fftw_malloc( pad*sizeof(fftw_complex) );
	double complex *f = (double complex*)out;
    double m = mean(data,n);
	for ( int i=0; i<pad; i++)
		in[i] = (i<n)? data[i]-m : 0;
	
    fftw_plan fwd, inv;

    #pragma omp critical(fftw_plan)
    {
        fwd = fftw_plan_dft_r2c_1d(pad, in, out, FFTW_ESTIMATE); //FFTW_MEASURE
        inv = fftw_plan_dft_c2r_1d(pad, out, in, FFTW_ESTIMATE);	
    }
    fftw_execute(fwd);
	for ( int k=0; k<pad/2 + 1; k++ )
		f[k] = f[k]*conj(f[k]);
    fftw_execute(inv);
	double c0 = in[0]/n;
    for ( int i = 0; i<n; i++ )
        in[i] /= (double)(n-i)*c0;
    
    #pragma omp critical(fftw_destroy)
    {
    fftw_destroy_plan(fwd);
    fftw_destroy_plan(inv);
    }
    fftw_free(out);
	return in;
}

double tau_int( double *data, int n, int sokal_c ) {
    double *restrict ac = acf_fft(data,n);
    double tau = 0.5;
    for (int k = 1; k < n; k++) {
        tau += ac[k];
        if ( k >= sokal_c*tau )
            break;
    }
    fftw_free(ac);
    return tau;
}

JackErr jackknife_block( double *data, int n, int bsize ) {
	int nbin = n/bsize;
	int samples = nbin*bsize;
	double mtot = 0, m2tot = 0, m4tot = 0;
	double norm = 1.0/((nbin-1)*bsize);
	
	double **restrict obs = malloc( 3*sizeof(double*) );
	for (int i=0; i<3; i++)
		obs[i] = malloc( nbin*sizeof(double) );

    for ( int i=0; i<samples; i++ ) {
		double d = data[i];
		double d2 = d*d;
        mtot += d;
        m2tot += d2;
        m4tot += d2*d2;
    }
    for ( int i=0; i<nbin; i++) {
		double m = mtot, m2 = m2tot, m4 = m4tot;
        for ( int j=0; j<bsize; j++) {
			double d = data[i*bsize + j];
			double d2 = d*d;
            m -= d;
            m2 -= d2;
            m4 -= d2*d2;
        }
        m *= norm;
        m2 *= norm;
        m4 *= norm;

        obs[0][i] = m;
        obs[1][i] = m2 - m*m;
		obs[2][i] = 1.0 - m4/(3.0*m2*m2);
    }
	JackErr err;
	double factor = (double)(nbin-1)*(nbin-1)/nbin;
	err.mean = sqrt( factor*variance(obs[0], nbin) );
	err.var = sqrt( factor*variance(obs[1], nbin) );
	err.bc = sqrt( factor*variance(obs[2], nbin) );

	for(int i=0; i<3; i++) free(obs[i]);
    free(obs);

	return err;
}

JackErr jackknife( double *data, int n )  {	
	JackErr err, max = {0};

	for ( int k=1; k<=n/20; k*=2 ) {	
		err = jackknife_block(data, n, k);
		max.mean = ( err.mean>max.mean ) ? err.mean : max.mean;
		max.var = ( err.var>max.var ) ? err.var : max.var;
		max.bc = ( err.bc>max.bc ) ? err.bc : max.bc;
        //printf("%d\t%d\t%lf\t%lf\t%lf\n", k, n/k, err.mean, err.var, err.bc);
	}
	return max;
}

double *load_betas( char *fname, int *nbetas ) {
    FILE *f = fopen(fname, "r");
    int i=0, count=0;
	double tmp;
    while ( fscanf(f, "%lf", &tmp) == 1 )
        count++;
    rewind(f); 
    double *betas = malloc( count * sizeof(double) );
    while ( fscanf(f, "%lf", &betas[i]) == 1 )
        i++;
    fclose(f);
    *nbetas = count;
    return betas;
}

double clocking( void ) {
	return clock()/(double)CLOCKS_PER_SEC;
}

//Xoshiro256plus by David Blackman and Sebastiano Vigna Unimi, https://prng.di.unimi.it
//Modified to generate double (float64)

uint64_t splitmix64( uint64_t *x ) {
	uint64_t z = (*x += 0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}

void seed_rng( RngState *state, int seed ) {
    uint64_t x = seed;
    state->s[0] = splitmix64(&x);
    state->s[1] = splitmix64(&x);
    state->s[2] = splitmix64(&x);
    state->s[3] = splitmix64(&x);
}

static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

uint64_t xoshiro256plus( RngState *state ) {
	const uint64_t result = state->s[0] + state->s[3];
	const uint64_t t = state->s[1] << 17;
	state->s[2] ^= state->s[0];
	state->s[3] ^= state->s[1];
	state->s[1] ^= state->s[2];
	state->s[0] ^= state->s[3];
	state->s[2] ^= t;
	state->s[3] = rotl(state->s[3], 45);
	return result;
}

static inline double rng( RngState *state ) {
	uint64_t x = xoshiro256plus(state);
	return (x >> 11) * 0x1.0p-53f;
}
