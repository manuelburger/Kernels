void star1(const int n, const int t, matrix & in, matrix & out) {
    auto inside = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1},{n-1,n-1},{t,t});
    Kokkos::parallel_for(inside, KOKKOS_LAMBDA(int i, int j) {
              out(i,j) += +in(i+-1,j+0) * -0.5
                          +in(i+0,j+-1) * -0.5
                          +in(i+0,j+1) * 0.5
                          +in(i+1,j+0) * 0.5;
    });
}

void star2(const int n, const int t, matrix & in, matrix & out) {
    auto inside = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({2,2},{n-2,n-2},{t,t});
    Kokkos::parallel_for(inside, KOKKOS_LAMBDA(int i, int j) {
              out(i,j) += +in(i+-2,j+0) * -0.125
                          +in(i+-1,j+0) * -0.25
                          +in(i+0,j+-2) * -0.125
                          +in(i+0,j+-1) * -0.25
                          +in(i+0,j+1) * 0.25
                          +in(i+0,j+2) * 0.125
                          +in(i+1,j+0) * 0.25
                          +in(i+2,j+0) * 0.125;
    });
}

void star3(const int n, const int t, matrix & in, matrix & out) {
    auto inside = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({3,3},{n-3,n-3},{t,t});
    Kokkos::parallel_for(inside, KOKKOS_LAMBDA(int i, int j) {
              out(i,j) += +in(i+-3,j+0) * -0.0555555555556
                          +in(i+-2,j+0) * -0.0833333333333
                          +in(i+-1,j+0) * -0.166666666667
                          +in(i+0,j+-3) * -0.0555555555556
                          +in(i+0,j+-2) * -0.0833333333333
                          +in(i+0,j+-1) * -0.166666666667
                          +in(i+0,j+1) * 0.166666666667
                          +in(i+0,j+2) * 0.0833333333333
                          +in(i+0,j+3) * 0.0555555555556
                          +in(i+1,j+0) * 0.166666666667
                          +in(i+2,j+0) * 0.0833333333333
                          +in(i+3,j+0) * 0.0555555555556;
    });
}

void star4(const int n, const int t, matrix & in, matrix & out) {
    auto inside = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({4,4},{n-4,n-4},{t,t});
    Kokkos::parallel_for(inside, KOKKOS_LAMBDA(int i, int j) {
              out(i,j) += +in(i+-4,j+0) * -0.03125
                          +in(i+-3,j+0) * -0.0416666666667
                          +in(i+-2,j+0) * -0.0625
                          +in(i+-1,j+0) * -0.125
                          +in(i+0,j+-4) * -0.03125
                          +in(i+0,j+-3) * -0.0416666666667
                          +in(i+0,j+-2) * -0.0625
                          +in(i+0,j+-1) * -0.125
                          +in(i+0,j+1) * 0.125
                          +in(i+0,j+2) * 0.0625
                          +in(i+0,j+3) * 0.0416666666667
                          +in(i+0,j+4) * 0.03125
                          +in(i+1,j+0) * 0.125
                          +in(i+2,j+0) * 0.0625
                          +in(i+3,j+0) * 0.0416666666667
                          +in(i+4,j+0) * 0.03125;
    });
}

void star5(const int n, const int t, matrix & in, matrix & out) {
    auto inside = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({5,5},{n-5,n-5},{t,t});
    Kokkos::parallel_for(inside, KOKKOS_LAMBDA(int i, int j) {
              out(i,j) += +in(i+-5,j+0) * -0.02
                          +in(i+-4,j+0) * -0.025
                          +in(i+-3,j+0) * -0.0333333333333
                          +in(i+-2,j+0) * -0.05
                          +in(i+-1,j+0) * -0.1
                          +in(i+0,j+-5) * -0.02
                          +in(i+0,j+-4) * -0.025
                          +in(i+0,j+-3) * -0.0333333333333
                          +in(i+0,j+-2) * -0.05
                          +in(i+0,j+-1) * -0.1
                          +in(i+0,j+1) * 0.1
                          +in(i+0,j+2) * 0.05
                          +in(i+0,j+3) * 0.0333333333333
                          +in(i+0,j+4) * 0.025
                          +in(i+0,j+5) * 0.02
                          +in(i+1,j+0) * 0.1
                          +in(i+2,j+0) * 0.05
                          +in(i+3,j+0) * 0.0333333333333
                          +in(i+4,j+0) * 0.025
                          +in(i+5,j+0) * 0.02;
    });
}

void grid1(const int n, const int t, matrix & in, matrix & out) {
    auto inside = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1},{n-1,n-1},{t,t});
    Kokkos::parallel_for(inside, KOKKOS_LAMBDA(int i, int j) {
              out(i,j) += +in(i+-1,j+-1) * -0.25
                          +in(i+-1,j+0) * -0.25
                          +in(i+0,j+-1) * -0.25
                          +in(i+0,j+1) * 0.25
                          +in(i+1,j+0) * 0.25
                          +in(i+1,j+1) * 0.25
                          ;
    });
}

void grid2(const int n, const int t, matrix & in, matrix & out) {
    auto inside = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({2,2},{n-2,n-2},{t,t});
    Kokkos::parallel_for(inside, KOKKOS_LAMBDA(int i, int j) {
              out(i,j) += +in(i+-2,j+-2) * -0.0625
                          +in(i+-2,j+-1) * -0.0208333333333
                          +in(i+-2,j+0) * -0.0208333333333
                          +in(i+-2,j+1) * -0.0208333333333
                          +in(i+-1,j+-2) * -0.0208333333333
                          +in(i+-1,j+-1) * -0.125
                          +in(i+-1,j+0) * -0.125
                          +in(i+-1,j+2) * 0.0208333333333
                          +in(i+0,j+-2) * -0.0208333333333
                          +in(i+0,j+-1) * -0.125
                          +in(i+0,j+1) * 0.125
                          +in(i+0,j+2) * 0.0208333333333
                          +in(i+1,j+-2) * -0.0208333333333
                          +in(i+1,j+0) * 0.125
                          +in(i+1,j+1) * 0.125
                          +in(i+1,j+2) * 0.0208333333333
                          +in(i+2,j+-1) * 0.0208333333333
                          +in(i+2,j+0) * 0.0208333333333
                          +in(i+2,j+1) * 0.0208333333333
                          +in(i+2,j+2) * 0.0625
                          ;
    });
}

void grid3(const int n, const int t, matrix & in, matrix & out) {
    auto inside = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({3,3},{n-3,n-3},{t,t});
    Kokkos::parallel_for(inside, KOKKOS_LAMBDA(int i, int j) {
              out(i,j) += +in(i+-3,j+-3) * -0.0277777777778
                          +in(i+-3,j+-2) * -0.00555555555556
                          +in(i+-3,j+-1) * -0.00555555555556
                          +in(i+-3,j+0) * -0.00555555555556
                          +in(i+-3,j+1) * -0.00555555555556
                          +in(i+-3,j+2) * -0.00555555555556
                          +in(i+-2,j+-3) * -0.00555555555556
                          +in(i+-2,j+-2) * -0.0416666666667
                          +in(i+-2,j+-1) * -0.0138888888889
                          +in(i+-2,j+0) * -0.0138888888889
                          +in(i+-2,j+1) * -0.0138888888889
                          +in(i+-2,j+3) * 0.00555555555556
                          +in(i+-1,j+-3) * -0.00555555555556
                          +in(i+-1,j+-2) * -0.0138888888889
                          +in(i+-1,j+-1) * -0.0833333333333
                          +in(i+-1,j+0) * -0.0833333333333
                          +in(i+-1,j+2) * 0.0138888888889
                          +in(i+-1,j+3) * 0.00555555555556
                          +in(i+0,j+-3) * -0.00555555555556
                          +in(i+0,j+-2) * -0.0138888888889
                          +in(i+0,j+-1) * -0.0833333333333
                          +in(i+0,j+1) * 0.0833333333333
                          +in(i+0,j+2) * 0.0138888888889
                          +in(i+0,j+3) * 0.00555555555556
                          +in(i+1,j+-3) * -0.00555555555556
                          +in(i+1,j+-2) * -0.0138888888889
                          +in(i+1,j+0) * 0.0833333333333
                          +in(i+1,j+1) * 0.0833333333333
                          +in(i+1,j+2) * 0.0138888888889
                          +in(i+1,j+3) * 0.00555555555556
                          +in(i+2,j+-3) * -0.00555555555556
                          +in(i+2,j+-1) * 0.0138888888889
                          +in(i+2,j+0) * 0.0138888888889
                          +in(i+2,j+1) * 0.0138888888889
                          +in(i+2,j+2) * 0.0416666666667
                          +in(i+2,j+3) * 0.00555555555556
                          +in(i+3,j+-2) * 0.00555555555556
                          +in(i+3,j+-1) * 0.00555555555556
                          +in(i+3,j+0) * 0.00555555555556
                          +in(i+3,j+1) * 0.00555555555556
                          +in(i+3,j+2) * 0.00555555555556
                          +in(i+3,j+3) * 0.0277777777778
                          ;
    });
}

void grid4(const int n, const int t, matrix & in, matrix & out) {
    auto inside = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({4,4},{n-4,n-4},{t,t});
    Kokkos::parallel_for(inside, KOKKOS_LAMBDA(int i, int j) {
              out(i,j) += +in(i+-4,j+-4) * -0.015625
                          +in(i+-4,j+-3) * -0.00223214285714
                          +in(i+-4,j+-2) * -0.00223214285714
                          +in(i+-4,j+-1) * -0.00223214285714
                          +in(i+-4,j+0) * -0.00223214285714
                          +in(i+-4,j+1) * -0.00223214285714
                          +in(i+-4,j+2) * -0.00223214285714
                          +in(i+-4,j+3) * -0.00223214285714
                          +in(i+-3,j+-4) * -0.00223214285714
                          +in(i+-3,j+-3) * -0.0208333333333
                          +in(i+-3,j+-2) * -0.00416666666667
                          +in(i+-3,j+-1) * -0.00416666666667
                          +in(i+-3,j+0) * -0.00416666666667
                          +in(i+-3,j+1) * -0.00416666666667
                          +in(i+-3,j+2) * -0.00416666666667
                          +in(i+-3,j+4) * 0.00223214285714
                          +in(i+-2,j+-4) * -0.00223214285714
                          +in(i+-2,j+-3) * -0.00416666666667
                          +in(i+-2,j+-2) * -0.03125
                          +in(i+-2,j+-1) * -0.0104166666667
                          +in(i+-2,j+0) * -0.0104166666667
                          +in(i+-2,j+1) * -0.0104166666667
                          +in(i+-2,j+3) * 0.00416666666667
                          +in(i+-2,j+4) * 0.00223214285714
                          +in(i+-1,j+-4) * -0.00223214285714
                          +in(i+-1,j+-3) * -0.00416666666667
                          +in(i+-1,j+-2) * -0.0104166666667
                          +in(i+-1,j+-1) * -0.0625
                          +in(i+-1,j+0) * -0.0625
                          +in(i+-1,j+2) * 0.0104166666667
                          +in(i+-1,j+3) * 0.00416666666667
                          +in(i+-1,j+4) * 0.00223214285714
                          +in(i+0,j+-4) * -0.00223214285714
                          +in(i+0,j+-3) * -0.00416666666667
                          +in(i+0,j+-2) * -0.0104166666667
                          +in(i+0,j+-1) * -0.0625
                          +in(i+0,j+1) * 0.0625
                          +in(i+0,j+2) * 0.0104166666667
                          +in(i+0,j+3) * 0.00416666666667
                          +in(i+0,j+4) * 0.00223214285714
                          +in(i+1,j+-4) * -0.00223214285714
                          +in(i+1,j+-3) * -0.00416666666667
                          +in(i+1,j+-2) * -0.0104166666667
                          +in(i+1,j+0) * 0.0625
                          +in(i+1,j+1) * 0.0625
                          +in(i+1,j+2) * 0.0104166666667
                          +in(i+1,j+3) * 0.00416666666667
                          +in(i+1,j+4) * 0.00223214285714
                          +in(i+2,j+-4) * -0.00223214285714
                          +in(i+2,j+-3) * -0.00416666666667
                          +in(i+2,j+-1) * 0.0104166666667
                          +in(i+2,j+0) * 0.0104166666667
                          +in(i+2,j+1) * 0.0104166666667
                          +in(i+2,j+2) * 0.03125
                          +in(i+2,j+3) * 0.00416666666667
                          +in(i+2,j+4) * 0.00223214285714
                          +in(i+3,j+-4) * -0.00223214285714
                          +in(i+3,j+-2) * 0.00416666666667
                          +in(i+3,j+-1) * 0.00416666666667
                          +in(i+3,j+0) * 0.00416666666667
                          +in(i+3,j+1) * 0.00416666666667
                          +in(i+3,j+2) * 0.00416666666667
                          +in(i+3,j+3) * 0.0208333333333
                          +in(i+3,j+4) * 0.00223214285714
                          +in(i+4,j+-3) * 0.00223214285714
                          +in(i+4,j+-2) * 0.00223214285714
                          +in(i+4,j+-1) * 0.00223214285714
                          +in(i+4,j+0) * 0.00223214285714
                          +in(i+4,j+1) * 0.00223214285714
                          +in(i+4,j+2) * 0.00223214285714
                          +in(i+4,j+3) * 0.00223214285714
                          +in(i+4,j+4) * 0.015625
                          ;
    });
}

void grid5(const int n, const int t, matrix & in, matrix & out) {
    auto inside = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({5,5},{n-5,n-5},{t,t});
    Kokkos::parallel_for(inside, KOKKOS_LAMBDA(int i, int j) {
              out(i,j) += +in(i+-5,j+-5) * -0.01
                          +in(i+-5,j+-4) * -0.00111111111111
                          +in(i+-5,j+-3) * -0.00111111111111
                          +in(i+-5,j+-2) * -0.00111111111111
                          +in(i+-5,j+-1) * -0.00111111111111
                          +in(i+-5,j+0) * -0.00111111111111
                          +in(i+-5,j+1) * -0.00111111111111
                          +in(i+-5,j+2) * -0.00111111111111
                          +in(i+-5,j+3) * -0.00111111111111
                          +in(i+-5,j+4) * -0.00111111111111
                          +in(i+-4,j+-5) * -0.00111111111111
                          +in(i+-4,j+-4) * -0.0125
                          +in(i+-4,j+-3) * -0.00178571428571
                          +in(i+-4,j+-2) * -0.00178571428571
                          +in(i+-4,j+-1) * -0.00178571428571
                          +in(i+-4,j+0) * -0.00178571428571
                          +in(i+-4,j+1) * -0.00178571428571
                          +in(i+-4,j+2) * -0.00178571428571
                          +in(i+-4,j+3) * -0.00178571428571
                          +in(i+-4,j+5) * 0.00111111111111
                          +in(i+-3,j+-5) * -0.00111111111111
                          +in(i+-3,j+-4) * -0.00178571428571
                          +in(i+-3,j+-3) * -0.0166666666667
                          +in(i+-3,j+-2) * -0.00333333333333
                          +in(i+-3,j+-1) * -0.00333333333333
                          +in(i+-3,j+0) * -0.00333333333333
                          +in(i+-3,j+1) * -0.00333333333333
                          +in(i+-3,j+2) * -0.00333333333333
                          +in(i+-3,j+4) * 0.00178571428571
                          +in(i+-3,j+5) * 0.00111111111111
                          +in(i+-2,j+-5) * -0.00111111111111
                          +in(i+-2,j+-4) * -0.00178571428571
                          +in(i+-2,j+-3) * -0.00333333333333
                          +in(i+-2,j+-2) * -0.025
                          +in(i+-2,j+-1) * -0.00833333333333
                          +in(i+-2,j+0) * -0.00833333333333
                          +in(i+-2,j+1) * -0.00833333333333
                          +in(i+-2,j+3) * 0.00333333333333
                          +in(i+-2,j+4) * 0.00178571428571
                          +in(i+-2,j+5) * 0.00111111111111
                          +in(i+-1,j+-5) * -0.00111111111111
                          +in(i+-1,j+-4) * -0.00178571428571
                          +in(i+-1,j+-3) * -0.00333333333333
                          +in(i+-1,j+-2) * -0.00833333333333
                          +in(i+-1,j+-1) * -0.05
                          +in(i+-1,j+0) * -0.05
                          +in(i+-1,j+2) * 0.00833333333333
                          +in(i+-1,j+3) * 0.00333333333333
                          +in(i+-1,j+4) * 0.00178571428571
                          +in(i+-1,j+5) * 0.00111111111111
                          +in(i+0,j+-5) * -0.00111111111111
                          +in(i+0,j+-4) * -0.00178571428571
                          +in(i+0,j+-3) * -0.00333333333333
                          +in(i+0,j+-2) * -0.00833333333333
                          +in(i+0,j+-1) * -0.05
                          +in(i+0,j+1) * 0.05
                          +in(i+0,j+2) * 0.00833333333333
                          +in(i+0,j+3) * 0.00333333333333
                          +in(i+0,j+4) * 0.00178571428571
                          +in(i+0,j+5) * 0.00111111111111
                          +in(i+1,j+-5) * -0.00111111111111
                          +in(i+1,j+-4) * -0.00178571428571
                          +in(i+1,j+-3) * -0.00333333333333
                          +in(i+1,j+-2) * -0.00833333333333
                          +in(i+1,j+0) * 0.05
                          +in(i+1,j+1) * 0.05
                          +in(i+1,j+2) * 0.00833333333333
                          +in(i+1,j+3) * 0.00333333333333
                          +in(i+1,j+4) * 0.00178571428571
                          +in(i+1,j+5) * 0.00111111111111
                          +in(i+2,j+-5) * -0.00111111111111
                          +in(i+2,j+-4) * -0.00178571428571
                          +in(i+2,j+-3) * -0.00333333333333
                          +in(i+2,j+-1) * 0.00833333333333
                          +in(i+2,j+0) * 0.00833333333333
                          +in(i+2,j+1) * 0.00833333333333
                          +in(i+2,j+2) * 0.025
                          +in(i+2,j+3) * 0.00333333333333
                          +in(i+2,j+4) * 0.00178571428571
                          +in(i+2,j+5) * 0.00111111111111
                          +in(i+3,j+-5) * -0.00111111111111
                          +in(i+3,j+-4) * -0.00178571428571
                          +in(i+3,j+-2) * 0.00333333333333
                          +in(i+3,j+-1) * 0.00333333333333
                          +in(i+3,j+0) * 0.00333333333333
                          +in(i+3,j+1) * 0.00333333333333
                          +in(i+3,j+2) * 0.00333333333333
                          +in(i+3,j+3) * 0.0166666666667
                          +in(i+3,j+4) * 0.00178571428571
                          +in(i+3,j+5) * 0.00111111111111
                          +in(i+4,j+-5) * -0.00111111111111
                          +in(i+4,j+-3) * 0.00178571428571
                          +in(i+4,j+-2) * 0.00178571428571
                          +in(i+4,j+-1) * 0.00178571428571
                          +in(i+4,j+0) * 0.00178571428571
                          +in(i+4,j+1) * 0.00178571428571
                          +in(i+4,j+2) * 0.00178571428571
                          +in(i+4,j+3) * 0.00178571428571
                          +in(i+4,j+4) * 0.0125
                          +in(i+4,j+5) * 0.00111111111111
                          +in(i+5,j+-4) * 0.00111111111111
                          +in(i+5,j+-3) * 0.00111111111111
                          +in(i+5,j+-2) * 0.00111111111111
                          +in(i+5,j+-1) * 0.00111111111111
                          +in(i+5,j+0) * 0.00111111111111
                          +in(i+5,j+1) * 0.00111111111111
                          +in(i+5,j+2) * 0.00111111111111
                          +in(i+5,j+3) * 0.00111111111111
                          +in(i+5,j+4) * 0.00111111111111
                          +in(i+5,j+5) * 0.01
                          ;
    });
}

