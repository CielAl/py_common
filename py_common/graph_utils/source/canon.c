#include "nauty/nausparse.h"
#include <stdint.h>
#include <assert.h>
#include <stdio.h>

int canonical(int num_edges,int num_node, const double *gr, const double *deg, const double*color, double* output) {

	// int ne = num_edges;
	// int n = num_node;

	DYNALLSTAT(int, orbits, orbits_sz);
	int lab[num_node];
	int ptn[num_node];

	for (int i = 0; i < num_node; ++i){
		lab[i] = i;
	}

	for (int i = 0; i < num_node; ++i){
		ptn[i] = (int)(color[i] + 0.5);
	}
	
	static DEFAULTOPTIONS_SPARSEGRAPH(options);
	statsblk stats;
	sparsegraph sg;
	sparsegraph canong;

	options.writeautoms = FALSE;
	options.getcanon = TRUE;
	options.defaultptn = FALSE;
	options.schreier = TRUE;

	SG_INIT(sg);
	SG_INIT(canong);
	int m = SETWORDSNEEDED(num_node);
	nauty_check(WORDSIZE,m,num_node,NAUTYVERSIONID);

	DYNALLOC1(int,orbits,orbits_sz,num_node,"malloc");
	
	SG_ALLOC(sg,num_node,num_edges,"malloc");
	sg.nv = num_node;
	sg.nde = num_edges;

	int degcount = 0;
	int k = 0;
	for (int i = 0; i < num_node; ++i){
		sg.v[i] = degcount;
		int rounded_deg = (int)(deg[i] + 0.5);
		sg.d[i] = rounded_deg;
		k = 0;
		// degree accumulation and check
		for (int j = 0; j < num_node; ++j){
			if ((int)(gr[i * num_node + j] + 0.5) == 0){ //gr[j * num_node + i]
			    continue;
			}
			else {
				sg.e[degcount + k] = j;
				++k;
			}
		}
		if(rounded_deg != k){
		    return 0;
		}
		degcount += k; 
	}

	sparsenauty(&sg, lab, ptn, orbits, &options, &stats, &canong);

	for (int i = 0; i < num_node; ++i){
		output[i] = (double) lab[i];
	}
	return 1;
}