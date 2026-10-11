// CPU-only probe of the very same header used by AthenaK and Athena++.
#include <fstream>
#include <vector>
#include "../codes/common/abel_problems.hpp"

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  const auto s = abel::ReadSpec(argv[1]);
  const int n = s.n;
  const size_t cells = static_cast<size_t>(n)*n*n;
  std::vector<double> out(cells*12, 0.0);
  for (int k=0; k<n; ++k) for (int j=0; j<n; ++j) for (int i=0; i<n; ++i) {
    size_t p = i + n*(j+n*k);
    auto c = abel::CellAverage(s,i,j,k);
    out[p] = c.rho;
    out[p+cells] = c.m[0]; out[p+2*cells] = c.m[1]; out[p+3*cells] = c.m[2];
    double magnetic=0;
    for (int d=0; d<3; ++d) {
      auto lo=abel::FaceB(s,d,i,j,k);
      auto hi=abel::FaceB(s,d,i+(d==0),j+(d==1),k+(d==2));
      double b=0.5*(lo+hi);
      out[p+(5+d)*cells]=b;
      out[p+(9+d)*cells]=lo;
      magnetic+=0.5*b*b;
    }
    out[p+4*cells]=c.ehydro+magnetic;
  }
  std::ofstream f(argv[2],std::ios::binary);
  f.write(reinterpret_cast<const char *>(out.data()),out.size()*sizeof(double));
  return f ? 0 : 1;
}
