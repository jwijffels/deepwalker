// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// embeddings_verse_pagerank
SEXP embeddings_verse_pagerank(Rcpp::IntegerVector dgrmatrix_p, Rcpp::IntegerVector dgrmatrix_j, int dimension, int epochs, float learning_rate, int samples, float alpha, int threads);
RcppExport SEXP _deepwalker_embeddings_verse_pagerank(SEXP dgrmatrix_pSEXP, SEXP dgrmatrix_jSEXP, SEXP dimensionSEXP, SEXP epochsSEXP, SEXP learning_rateSEXP, SEXP samplesSEXP, SEXP alphaSEXP, SEXP threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type dgrmatrix_p(dgrmatrix_pSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type dgrmatrix_j(dgrmatrix_jSEXP);
    Rcpp::traits::input_parameter< int >::type dimension(dimensionSEXP);
    Rcpp::traits::input_parameter< int >::type epochs(epochsSEXP);
    Rcpp::traits::input_parameter< float >::type learning_rate(learning_rateSEXP);
    Rcpp::traits::input_parameter< int >::type samples(samplesSEXP);
    Rcpp::traits::input_parameter< float >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< int >::type threads(threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(embeddings_verse_pagerank(dgrmatrix_p, dgrmatrix_j, dimension, epochs, learning_rate, samples, alpha, threads));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_deepwalker_embeddings_verse_pagerank", (DL_FUNC) &_deepwalker_embeddings_verse_pagerank, 8},
    {NULL, NULL, 0}
};

RcppExport void R_init_deepwalker(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
