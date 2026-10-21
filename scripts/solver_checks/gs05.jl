# Edge EMF from the four face EMFs around an edge, upwinded by the mass flux (Gardiner & Stone 2005, eqs. 41-45,
# "CT-contact"). The edge sits at (a-1/2, b-1/2) in the plane of the two directions a, b across it; Ea(db) is the EMF
# on the a-face (a-1/2, b+db), Eb(da) on the b-face (a+da, b-1/2), mA/mB the mass fluxes through those faces, and
# Er(da,db) the cell-centred EMF of cell (a+da, b+db), all for db, da in {0, -1}. With all derivative terms equal it
# is the plain average of the four face EMFs.
@inline function gs05(Ea0, Eam, Eb0, Ebm, mA0, mAm, mB0, mBm, Er00, Erm0, Er0m, Ermm)
    up(m, lower, upper) = m > 0 ? lower : (m < 0 ? upper : (lower + upper)/2)   # lower: the side at -1 (upwind for m > 0)
    A1 = up(mA0, Erm0 - Ebm, Er00 - Eb0)      # dE/db between the cell centres of row b and the b-face below them
    A3 = up(mAm, Ebm - Ermm, Eb0 - Er0m)      # dE/db between the b-face and the cell centres of row b-1
    B1 = up(mB0, Er0m - Eam, Er00 - Ea0)      # dE/da between the cell centres of column a and the a-face before them
    B3 = up(mBm, Eam - Ermm, Ea0 - Erm0)      # dE/da between the a-face and the cell centres of column a-1
    ((Ea0 + Eam + Eb0 + Ebm) + (A3 - A1) + (B3 - B1))/4
end
