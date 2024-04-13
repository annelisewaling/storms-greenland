;**************************************
;var_cam.ncl
;**************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"
load "/glade/work/aherring/CESM2/arh-git-scripts/ncl/se_func_lib.ncl"
;**************************************
begin
  p0 = 1000.d2
  secpday = 60.*60.*24;
  pi = 3.14159265358979
  g = 9.80616
  Re = 6371.22e3
  cp = 1004.
  Rd = 287.
;****************************************

  subdir_scrip = "../data/expand/"

  nmodel = 8
  ndist = 13
  fname0 = new( (/ndist,nmodel/), "string")
  fname1 = new( (/ndist,nmodel/), "string")
  fname2 = new( (/ndist,nmodel/), "string")
  fname3 = new( (/ndist,nmodel/), "string")     

;;---Mean

fname0(:,0) = (/"geolines_PRECT_amean_pm1day_f19.nc", \
	        "geolines_PRECT_amean_exp100km_pm1day_f19.nc", \
                "geolines_PRECT_amean_exp200km_pm1day_f19.nc", \
                "geolines_PRECT_amean_exp300km_pm1day_f19.nc", \
                "geolines_PRECT_amean_exp400km_pm1day_f19.nc", \
                "geolines_PRECT_amean_exp500km_pm1day_f19.nc", \
                "geolines_PRECT_amean_exp600km_pm1day_f19.nc", \
                "geolines_PRECT_amean_exp700km_pm1day_f19.nc", \
                "geolines_PRECT_amean_exp800km_pm1day_f19.nc", \
                "geolines_PRECT_amean_exp900km_pm1day_f19.nc", \
                "geolines_PRECT_amean_exp1000km_pm1day_f19.nc", \
                "geolines_PRECT_amean_exp1100km_pm1day_f19.nc", \
                "geolines_PRECT_amean_exp1200km_pm1day_f19.nc"/)

fname0(:,1) = (/"geolines_PRECT_amean_pm1day_f09.nc", \
                "geolines_PRECT_amean_exp100km_pm1day_f09.nc", \
                "geolines_PRECT_amean_exp200km_pm1day_f09.nc", \
                "geolines_PRECT_amean_exp300km_pm1day_f09.nc", \
                "geolines_PRECT_amean_exp400km_pm1day_f09.nc", \
                "geolines_PRECT_amean_exp500km_pm1day_f09.nc", \
                "geolines_PRECT_amean_exp600km_pm1day_f09.nc", \
                "geolines_PRECT_amean_exp700km_pm1day_f09.nc", \
                "geolines_PRECT_amean_exp800km_pm1day_f09.nc", \
                "geolines_PRECT_amean_exp900km_pm1day_f09.nc", \
                "geolines_PRECT_amean_exp1000km_pm1day_f09.nc", \
                "geolines_PRECT_amean_exp1100km_pm1day_f09.nc", \
                "geolines_PRECT_amean_exp1200km_pm1day_f09.nc"/)

fname0(:,2) = (/"geolines_PRECT_amean_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_exp100km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_exp200km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_exp300km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_exp400km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_exp500km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_exp600km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_exp700km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_exp800km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_exp900km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_exp1000km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_exp1100km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_exp1200km_pm1day_ne30pg2.nc"/)

fname0(:,3) = (/"geolines_PRECT_amean_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_exp100km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_exp200km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_exp300km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_exp400km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_exp500km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_exp600km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_exp700km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_exp800km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_exp900km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_exp1000km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_exp1100km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_exp1200km_pm1day_ne30pg3.nc"/)

fname0(:,4) = (/"geolines_PRECT_amean_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_exp100km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_exp200km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_exp300km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_exp400km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_exp500km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_exp600km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_exp700km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_exp800km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_exp900km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_exp1000km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_exp1100km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_exp1200km_pm1day_ARCTIC.nc"/)

fname0(:,5) = (/"geolines_PRECT_amean_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_exp100km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_exp200km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_exp300km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_exp400km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_exp500km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_exp600km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_exp700km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_exp800km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_exp900km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_exp1000km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_exp1100km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_exp1200km_pm1day_ARCTICGRIS.nc"/)

fname0(:,6) = (/"geolines_PRECT_amean_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_exp100km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_exp200km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_exp300km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_exp400km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_exp500km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_exp600km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_exp700km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_exp800km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_exp900km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_exp1000km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_exp1100km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_exp1200km_pm1day_ERA5.nc"/)

fname0(:,7) = (/"geolines_PRECT_amean_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_exp100km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_exp200km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_exp300km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_exp400km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_exp500km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_exp600km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_exp700km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_exp800km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_exp900km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_exp1000km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_exp1100km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_exp1200km_pm1day_MERRA2.nc"/)

;;---95th percentile

fname1(:,0) = (/"geolines_PRECT_amean_95ptile_f19.nc", \
	        "geolines_PRECT_amean_95ptile_exp100km_pm1day_f19.nc", \
                "geolines_PRECT_amean_95ptile_exp200km_pm1day_f19.nc", \
                "geolines_PRECT_amean_95ptile_exp300km_pm1day_f19.nc", \
                "geolines_PRECT_amean_95ptile_exp400km_pm1day_f19.nc", \
                "geolines_PRECT_amean_95ptile_exp500km_pm1day_f19.nc", \
                "geolines_PRECT_amean_95ptile_exp600km_pm1day_f19.nc", \
                "geolines_PRECT_amean_95ptile_exp700km_pm1day_f19.nc", \
                "geolines_PRECT_amean_95ptile_exp800km_pm1day_f19.nc", \
                "geolines_PRECT_amean_95ptile_exp900km_pm1day_f19.nc", \
                "geolines_PRECT_amean_95ptile_exp1000km_pm1day_f19.nc", \
                "geolines_PRECT_amean_95ptile_exp1100km_pm1day_f19.nc", \
                "geolines_PRECT_amean_95ptile_exp1200km_pm1day_f19.nc"/)

;nxtra = 0
;tname1 = (/"geolines_PRECT_amean_ravg_95ptile_pm1day_f19.nc", \
;                "geolines_PRECT_amean_ravg_95ptile_exp100km_pm1day_f19.nc", \
;                "geolines_PRECT_amean_ravg_95ptile_exp200km_pm1day_f19.nc", \
;                "geolines_PRECT_amean_ravg_95ptile_exp300km_pm1day_f19.nc", \
;                "geolines_PRECT_amean_ravg_95ptile_exp500km_pm1day_f19.nc", \
;                "geolines_PRECT_amean_ravg_95ptile_exp500km_pm1day_f19.nc", \
;                "geolines_PRECT_amean_ravg_95ptile_exp600km_pm1day_f19.nc", \
;                "geolines_PRECT_amean_ravg_95ptile_exp700km_pm1day_f19.nc", \
;                "geolines_PRECT_amean_ravg_95ptile_exp800km_pm1day_f19.nc", \
;                "geolines_PRECT_amean_ravg_95ptile_exp900km_pm1day_f19.nc", \
;                "geolines_PRECT_amean_ravg_95ptile_exp1000km_pm1day_f19.nc", \
;                "geolines_PRECT_amean_ravg_95ptile_exp1100km_pm1day_f19.nc", \
;                "geolines_PRECT_amean_ravg_95ptile_exp1200km_pm1day_f19.nc"/)


fname1(:,1) = (/"geolines_PRECT_amean_95ptile_f09.nc", \
                "geolines_PRECT_amean_95ptile_exp100km_pm1day_f09.nc", \
                "geolines_PRECT_amean_95ptile_exp200km_pm1day_f09.nc", \
                "geolines_PRECT_amean_95ptile_exp300km_pm1day_f09.nc", \
                "geolines_PRECT_amean_95ptile_exp400km_pm1day_f09.nc", \
                "geolines_PRECT_amean_95ptile_exp500km_pm1day_f09.nc", \
                "geolines_PRECT_amean_95ptile_exp600km_pm1day_f09.nc", \
                "geolines_PRECT_amean_95ptile_exp700km_pm1day_f09.nc", \
                "geolines_PRECT_amean_95ptile_exp800km_pm1day_f09.nc", \
                "geolines_PRECT_amean_95ptile_exp900km_pm1day_f09.nc", \
                "geolines_PRECT_amean_95ptile_exp1000km_pm1day_f09.nc", \
                "geolines_PRECT_amean_95ptile_exp1100km_pm1day_f09.nc", \
                "geolines_PRECT_amean_95ptile_exp1200km_pm1day_f09.nc"/)

fname1(:,2) = (/"geolines_PRECT_amean_95ptile_ne30pg2.nc", \
                "geolines_PRECT_amean_95ptile_exp100km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_95ptile_exp200km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_95ptile_exp300km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_95ptile_exp400km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_95ptile_exp500km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_95ptile_exp600km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_95ptile_exp700km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_95ptile_exp800km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_95ptile_exp900km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_95ptile_exp1000km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_95ptile_exp1100km_pm1day_ne30pg2.nc", \
                "geolines_PRECT_amean_95ptile_exp1200km_pm1day_ne30pg2.nc"/)

fname1(:,3) = (/"geolines_PRECT_amean_95ptile_ne30pg3.nc", \
                "geolines_PRECT_amean_95ptile_exp100km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_95ptile_exp200km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_95ptile_exp300km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_95ptile_exp400km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_95ptile_exp500km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_95ptile_exp600km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_95ptile_exp700km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_95ptile_exp800km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_95ptile_exp900km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_95ptile_exp1000km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_95ptile_exp1100km_pm1day_ne30pg3.nc", \
                "geolines_PRECT_amean_95ptile_exp1200km_pm1day_ne30pg3.nc"/)

fname1(:,4) = (/"geolines_PRECT_amean_95ptile_ARCTIC.nc", \
                "geolines_PRECT_amean_95ptile_exp100km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_95ptile_exp200km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_95ptile_exp300km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_95ptile_exp400km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_95ptile_exp500km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_95ptile_exp600km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_95ptile_exp700km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_95ptile_exp800km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_95ptile_exp900km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_95ptile_exp1000km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_95ptile_exp1100km_pm1day_ARCTIC.nc", \
                "geolines_PRECT_amean_95ptile_exp1200km_pm1day_ARCTIC.nc"/)

nxtra = 4       
tname1 = (/"geolines_PRECT_amean_ravg_95ptile_pm1day_ARCTIC.nc", \
           "geolines_PRECT_amean_ravg_95ptile_exp100km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_amean_ravg_95ptile_exp200km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_amean_ravg_95ptile_exp300km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_amean_ravg_95ptile_exp400km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_amean_ravg_95ptile_exp500km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_amean_ravg_95ptile_exp600km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_amean_ravg_95ptile_exp700km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_amean_ravg_95ptile_exp800km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_amean_ravg_95ptile_exp900km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_amean_ravg_95ptile_exp1000km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_amean_ravg_95ptile_exp1100km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_amean_ravg_95ptile_exp1200km_pm1day_ARCTIC.nc"/)

fname1(:,5) = (/"geolines_PRECT_amean_95ptile_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_95ptile_exp100km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_95ptile_exp200km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_95ptile_exp300km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_95ptile_exp400km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_95ptile_exp500km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_95ptile_exp600km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_95ptile_exp700km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_95ptile_exp800km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_95ptile_exp900km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_95ptile_exp1000km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_95ptile_exp1100km_pm1day_ARCTICGRIS.nc", \
                "geolines_PRECT_amean_95ptile_exp1200km_pm1day_ARCTICGRIS.nc"/)

fname1(:,6) = (/"geolines_PRECT_amean_95ptile_ERA5.nc", \
                "geolines_PRECT_amean_95ptile_exp100km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_95ptile_exp200km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_95ptile_exp300km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_95ptile_exp400km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_95ptile_exp500km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_95ptile_exp600km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_95ptile_exp700km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_95ptile_exp800km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_95ptile_exp900km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_95ptile_exp1000km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_95ptile_exp1100km_pm1day_MERRA2.nc", \
                "geolines_PRECT_amean_95ptile_exp1200km_pm1day_MERRA2.nc"/)

fname1(:,7) = (/"geolines_PRECT_amean_95ptile_MERRA2.nc", \
                "geolines_PRECT_amean_95ptile_exp100km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_95ptile_exp200km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_95ptile_exp300km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_95ptile_exp400km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_95ptile_exp500km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_95ptile_exp600km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_95ptile_exp700km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_95ptile_exp800km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_95ptile_exp900km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_95ptile_exp1000km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_95ptile_exp1100km_pm1day_ERA5.nc", \
                "geolines_PRECT_amean_95ptile_exp1200km_pm1day_ERA5.nc"/)

;;---Integrated

fname2(:,0) = (/"geolines_PRECT_aint_pm1day_f19.nc", \
	   "geolines_PRECT_aint_exp100km_pm1day_f19.nc", \
           "geolines_PRECT_aint_exp200km_pm1day_f19.nc", \
           "geolines_PRECT_aint_exp300km_pm1day_f19.nc", \
           "geolines_PRECT_aint_exp400km_pm1day_f19.nc", \
           "geolines_PRECT_aint_exp500km_pm1day_f19.nc", \
           "geolines_PRECT_aint_exp600km_pm1day_f19.nc", \
           "geolines_PRECT_aint_exp700km_pm1day_f19.nc", \
           "geolines_PRECT_aint_exp800km_pm1day_f19.nc", \
           "geolines_PRECT_aint_exp900km_pm1day_f19.nc", \
           "geolines_PRECT_aint_exp1000km_pm1day_f19.nc", \
           "geolines_PRECT_aint_exp1100km_pm1day_f19.nc", \
           "geolines_PRECT_aint_exp1200km_pm1day_f19.nc"/)

fname2(:,1) = (/"geolines_PRECT_aint_pm1day_f09.nc", \
           "geolines_PRECT_aint_exp100km_pm1day_f09.nc", \
           "geolines_PRECT_aint_exp200km_pm1day_f09.nc", \
           "geolines_PRECT_aint_exp300km_pm1day_f09.nc", \
           "geolines_PRECT_aint_exp400km_pm1day_f09.nc", \
           "geolines_PRECT_aint_exp500km_pm1day_f09.nc", \
           "geolines_PRECT_aint_exp600km_pm1day_f09.nc", \
           "geolines_PRECT_aint_exp700km_pm1day_f09.nc", \
           "geolines_PRECT_aint_exp800km_pm1day_f09.nc", \
           "geolines_PRECT_aint_exp900km_pm1day_f09.nc", \
           "geolines_PRECT_aint_exp1000km_pm1day_f09.nc", \
           "geolines_PRECT_aint_exp1100km_pm1day_f09.nc", \
           "geolines_PRECT_aint_exp1200km_pm1day_f09.nc"/)

fname2(:,2) = (/"geolines_PRECT_aint_pm1day_ne30pg2.nc", \
           "geolines_PRECT_aint_exp100km_pm1day_ne30pg2.nc", \
           "geolines_PRECT_aint_exp200km_pm1day_ne30pg2.nc", \
           "geolines_PRECT_aint_exp300km_pm1day_ne30pg2.nc", \
           "geolines_PRECT_aint_exp400km_pm1day_ne30pg2.nc", \
           "geolines_PRECT_aint_exp500km_pm1day_ne30pg2.nc", \
           "geolines_PRECT_aint_exp600km_pm1day_ne30pg2.nc", \
           "geolines_PRECT_aint_exp700km_pm1day_ne30pg2.nc", \
           "geolines_PRECT_aint_exp800km_pm1day_ne30pg2.nc", \
           "geolines_PRECT_aint_exp900km_pm1day_ne30pg2.nc", \
           "geolines_PRECT_aint_exp1000km_pm1day_ne30pg2.nc", \
           "geolines_PRECT_aint_exp1100km_pm1day_ne30pg2.nc", \
           "geolines_PRECT_aint_exp1200km_pm1day_ne30pg2.nc"/)

fname2(:,3) = (/"geolines_PRECT_aint_pm1day_ne30pg3.nc", \
           "geolines_PRECT_aint_exp100km_pm1day_ne30pg3.nc", \
           "geolines_PRECT_aint_exp200km_pm1day_ne30pg3.nc", \
           "geolines_PRECT_aint_exp300km_pm1day_ne30pg3.nc", \
           "geolines_PRECT_aint_exp400km_pm1day_ne30pg3.nc", \
           "geolines_PRECT_aint_exp500km_pm1day_ne30pg3.nc", \
           "geolines_PRECT_aint_exp600km_pm1day_ne30pg3.nc", \
           "geolines_PRECT_aint_exp700km_pm1day_ne30pg3.nc", \
           "geolines_PRECT_aint_exp800km_pm1day_ne30pg3.nc", \
           "geolines_PRECT_aint_exp900km_pm1day_ne30pg3.nc", \
           "geolines_PRECT_aint_exp1000km_pm1day_ne30pg3.nc", \
           "geolines_PRECT_aint_exp1100km_pm1day_ne30pg3.nc", \
           "geolines_PRECT_aint_exp1200km_pm1day_ne30pg3.nc"/)

fname2(:,4) = (/"geolines_PRECT_aint_pm1day_ARCTIC.nc", \
           "geolines_PRECT_aint_exp100km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_aint_exp200km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_aint_exp300km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_aint_exp400km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_aint_exp500km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_aint_exp600km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_aint_exp700km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_aint_exp800km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_aint_exp900km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_aint_exp1000km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_aint_exp1100km_pm1day_ARCTIC.nc", \
           "geolines_PRECT_aint_exp1200km_pm1day_ARCTIC.nc"/)

fname2(:,5) = (/"geolines_PRECT_aint_pm1day_ARCTICGRIS.nc", \
           "geolines_PRECT_aint_exp100km_pm1day_ARCTICGRIS.nc", \
           "geolines_PRECT_aint_exp200km_pm1day_ARCTICGRIS.nc", \
           "geolines_PRECT_aint_exp300km_pm1day_ARCTICGRIS.nc", \
           "geolines_PRECT_aint_exp400km_pm1day_ARCTICGRIS.nc", \
           "geolines_PRECT_aint_exp500km_pm1day_ARCTICGRIS.nc", \
           "geolines_PRECT_aint_exp600km_pm1day_ARCTICGRIS.nc", \
           "geolines_PRECT_aint_exp700km_pm1day_ARCTICGRIS.nc", \
           "geolines_PRECT_aint_exp800km_pm1day_ARCTICGRIS.nc", \
           "geolines_PRECT_aint_exp900km_pm1day_ARCTICGRIS.nc", \
           "geolines_PRECT_aint_exp1000km_pm1day_ARCTICGRIS.nc", \
           "geolines_PRECT_aint_exp1100km_pm1day_ARCTICGRIS.nc", \
           "geolines_PRECT_aint_exp1200km_pm1day_ARCTICGRIS.nc"/)

fname2(:,6) = (/"geolines_PRECT_aint_pm1day_ERA5.nc", \
           "geolines_PRECT_aint_exp100km_pm1day_ERA5.nc", \
           "geolines_PRECT_aint_exp200km_pm1day_ERA5.nc", \
           "geolines_PRECT_aint_exp300km_pm1day_ERA5.nc", \
           "geolines_PRECT_aint_exp400km_pm1day_ERA5.nc", \
           "geolines_PRECT_aint_exp500km_pm1day_ERA5.nc", \
           "geolines_PRECT_aint_exp600km_pm1day_ERA5.nc", \
           "geolines_PRECT_aint_exp700km_pm1day_ERA5.nc", \
           "geolines_PRECT_aint_exp800km_pm1day_ERA5.nc", \
           "geolines_PRECT_aint_exp900km_pm1day_ERA5.nc", \
           "geolines_PRECT_aint_exp1000km_pm1day_ERA5.nc", \
           "geolines_PRECT_aint_exp1100km_pm1day_ERA5.nc", \
           "geolines_PRECT_aint_exp1200km_pm1day_ERA5.nc"/)

fname2(:,7) = (/"geolines_PRECT_aint_pm1day_MERRA2.nc", \
           "geolines_PRECT_aint_exp100km_pm1day_MERRA2.nc", \
           "geolines_PRECT_aint_exp200km_pm1day_MERRA2.nc", \
           "geolines_PRECT_aint_exp300km_pm1day_MERRA2.nc", \
           "geolines_PRECT_aint_exp400km_pm1day_MERRA2.nc", \
           "geolines_PRECT_aint_exp500km_pm1day_MERRA2.nc", \
           "geolines_PRECT_aint_exp600km_pm1day_MERRA2.nc", \
           "geolines_PRECT_aint_exp700km_pm1day_MERRA2.nc", \
           "geolines_PRECT_aint_exp800km_pm1day_MERRA2.nc", \
           "geolines_PRECT_aint_exp900km_pm1day_MERRA2.nc", \
           "geolines_PRECT_aint_exp1000km_pm1day_MERRA2.nc", \
           "geolines_PRECT_aint_exp1100km_pm1day_MERRA2.nc", \
           "geolines_PRECT_aint_exp1200km_pm1day_MERRA2.nc"/)

;---95th Percentile

fname3(:,0) = (/"geolines_PRECT_aint_95ptile_f19.nc", \
	       "geolines_PRECT_aint_95ptile_exp100km_pm1day_f19.nc", \
               "geolines_PRECT_aint_95ptile_exp200km_pm1day_f19.nc", \
               "geolines_PRECT_aint_95ptile_exp300km_pm1day_f19.nc", \
               "geolines_PRECT_aint_95ptile_exp400km_pm1day_f19.nc", \
               "geolines_PRECT_aint_95ptile_exp500km_pm1day_f19.nc", \
               "geolines_PRECT_aint_95ptile_exp600km_pm1day_f19.nc", \
               "geolines_PRECT_aint_95ptile_exp700km_pm1day_f19.nc", \
               "geolines_PRECT_aint_95ptile_exp800km_pm1day_f19.nc", \
               "geolines_PRECT_aint_95ptile_exp900km_pm1day_f19.nc", \
               "geolines_PRECT_aint_95ptile_exp1000km_pm1day_f19.nc", \
               "geolines_PRECT_aint_95ptile_exp1100km_pm1day_f19.nc", \
               "geolines_PRECT_aint_95ptile_exp1200km_pm1day_f19.nc"/)
     
fname3(:,1) = (/"geolines_PRECT_aint_95ptile_f09.nc", \
	       "geolines_PRECT_aint_95ptile_exp100km_pm1day_f09.nc", \
               "geolines_PRECT_aint_95ptile_exp200km_pm1day_f09.nc", \
               "geolines_PRECT_aint_95ptile_exp300km_pm1day_f09.nc", \
               "geolines_PRECT_aint_95ptile_exp400km_pm1day_f09.nc", \
               "geolines_PRECT_aint_95ptile_exp500km_pm1day_f09.nc", \
               "geolines_PRECT_aint_95ptile_exp600km_pm1day_f09.nc", \
               "geolines_PRECT_aint_95ptile_exp700km_pm1day_f09.nc", \
               "geolines_PRECT_aint_95ptile_exp800km_pm1day_f09.nc", \
               "geolines_PRECT_aint_95ptile_exp900km_pm1day_f09.nc", \
               "geolines_PRECT_aint_95ptile_exp1000km_pm1day_f09.nc", \
               "geolines_PRECT_aint_95ptile_exp1100km_pm1day_f09.nc", \
               "geolines_PRECT_aint_95ptile_exp1200km_pm1day_f09.nc"/)

fname3(:,2) = (/"geolines_PRECT_aint_95ptile_ne30pg2.nc", \
	       "geolines_PRECT_aint_95ptile_exp100km_pm1day_ne30pg2.nc", \
               "geolines_PRECT_aint_95ptile_exp200km_pm1day_ne30pg2.nc", \
               "geolines_PRECT_aint_95ptile_exp300km_pm1day_ne30pg2.nc", \
               "geolines_PRECT_aint_95ptile_exp400km_pm1day_ne30pg2.nc", \
               "geolines_PRECT_aint_95ptile_exp500km_pm1day_ne30pg2.nc", \
               "geolines_PRECT_aint_95ptile_exp600km_pm1day_ne30pg2.nc", \
               "geolines_PRECT_aint_95ptile_exp700km_pm1day_ne30pg2.nc", \
               "geolines_PRECT_aint_95ptile_exp800km_pm1day_ne30pg2.nc", \
               "geolines_PRECT_aint_95ptile_exp900km_pm1day_ne30pg2.nc", \
               "geolines_PRECT_aint_95ptile_exp1000km_pm1day_ne30pg2.nc", \
               "geolines_PRECT_aint_95ptile_exp1100km_pm1day_ne30pg2.nc", \
               "geolines_PRECT_aint_95ptile_exp1200km_pm1day_ne30pg2.nc"/)          
 
fname3(:,3) = (/"geolines_PRECT_aint_95ptile_ne30pg3.nc", \ 
	       "geolines_PRECT_aint_95ptile_exp100km_pm1day_ne30pg3.nc", \
               "geolines_PRECT_aint_95ptile_exp200km_pm1day_ne30pg3.nc", \
               "geolines_PRECT_aint_95ptile_exp300km_pm1day_ne30pg3.nc", \
               "geolines_PRECT_aint_95ptile_exp400km_pm1day_ne30pg3.nc", \
               "geolines_PRECT_aint_95ptile_exp500km_pm1day_ne30pg3.nc", \
               "geolines_PRECT_aint_95ptile_exp600km_pm1day_ne30pg3.nc", \
               "geolines_PRECT_aint_95ptile_exp700km_pm1day_ne30pg3.nc", \
               "geolines_PRECT_aint_95ptile_exp800km_pm1day_ne30pg3.nc", \
               "geolines_PRECT_aint_95ptile_exp900km_pm1day_ne30pg3.nc", \
               "geolines_PRECT_aint_95ptile_exp1000km_pm1day_ne30pg3.nc", \
               "geolines_PRECT_aint_95ptile_exp1100km_pm1day_ne30pg3.nc", \
               "geolines_PRECT_aint_95ptile_exp1200km_pm1day_ne30pg3.nc"/)

fname3(:,4) = (/"geolines_PRECT_aint_95ptile_ARCTIC.nc", \
	       "geolines_PRECT_aint_95ptile_exp100km_pm1day_ARCTIC.nc", \
               "geolines_PRECT_aint_95ptile_exp200km_pm1day_ARCTIC.nc", \
               "geolines_PRECT_aint_95ptile_exp300km_pm1day_ARCTIC.nc", \
               "geolines_PRECT_aint_95ptile_exp400km_pm1day_ARCTIC.nc", \
               "geolines_PRECT_aint_95ptile_exp500km_pm1day_ARCTIC.nc", \
               "geolines_PRECT_aint_95ptile_exp600km_pm1day_ARCTIC.nc", \
               "geolines_PRECT_aint_95ptile_exp700km_pm1day_ARCTIC.nc", \
               "geolines_PRECT_aint_95ptile_exp800km_pm1day_ARCTIC.nc", \
               "geolines_PRECT_aint_95ptile_exp900km_pm1day_ARCTIC.nc", \
               "geolines_PRECT_aint_95ptile_exp1000km_pm1day_ARCTIC.nc", \
               "geolines_PRECT_aint_95ptile_exp1100km_pm1day_ARCTIC.nc", \
               "geolines_PRECT_aint_95ptile_exp1200km_pm1day_ARCTIC.nc"/)

fname3(:,5) = (/"geolines_PRECT_aint_95ptile_ARCTICGRIS.nc", \
	       "geolines_PRECT_aint_95ptile_exp100km_pm1day_ARCTICGRIS.nc", \
               "geolines_PRECT_aint_95ptile_exp200km_pm1day_ARCTICGRIS.nc", \
               "geolines_PRECT_aint_95ptile_exp300km_pm1day_ARCTICGRIS.nc", \
               "geolines_PRECT_aint_95ptile_exp400km_pm1day_ARCTICGRIS.nc", \
               "geolines_PRECT_aint_95ptile_exp500km_pm1day_ARCTICGRIS.nc", \
               "geolines_PRECT_aint_95ptile_exp600km_pm1day_ARCTICGRIS.nc", \
               "geolines_PRECT_aint_95ptile_exp700km_pm1day_ARCTICGRIS.nc", \
               "geolines_PRECT_aint_95ptile_exp800km_pm1day_ARCTICGRIS.nc", \
               "geolines_PRECT_aint_95ptile_exp900km_pm1day_ARCTICGRIS.nc", \
               "geolines_PRECT_aint_95ptile_exp1000km_pm1day_ARCTICGRIS.nc", \
               "geolines_PRECT_aint_95ptile_exp1100km_pm1day_ARCTICGRIS.nc", \
               "geolines_PRECT_aint_95ptile_exp1200km_pm1day_ARCTICGRIS.nc"/)

fname3(:,6) = (/"geolines_PRECT_aint_95ptile_ERA5.nc", \
	       "geolines_PRECT_aint_95ptile_exp100km_pm1day_ERA5.nc", \
               "geolines_PRECT_aint_95ptile_exp200km_pm1day_ERA5.nc", \
               "geolines_PRECT_aint_95ptile_exp300km_pm1day_ERA5.nc", \
               "geolines_PRECT_aint_95ptile_exp400km_pm1day_ERA5.nc", \
               "geolines_PRECT_aint_95ptile_exp500km_pm1day_ERA5.nc", \
               "geolines_PRECT_aint_95ptile_exp600km_pm1day_ERA5.nc", \
               "geolines_PRECT_aint_95ptile_exp700km_pm1day_ERA5.nc", \
               "geolines_PRECT_aint_95ptile_exp800km_pm1day_ERA5.nc", \
               "geolines_PRECT_aint_95ptile_exp900km_pm1day_ERA5.nc", \
               "geolines_PRECT_aint_95ptile_exp1000km_pm1day_ERA5.nc", \
               "geolines_PRECT_aint_95ptile_exp1100km_pm1day_ERA5.nc", \
               "geolines_PRECT_aint_95ptile_exp1200km_pm1day_ERA5.nc"/)

fname3(:,7) = (/"geolines_PRECT_aint_95ptile_MERRA2.nc", \
	       "geolines_PRECT_aint_95ptile_exp100km_pm1day_MERRA2.nc", \
	       "geolines_PRECT_aint_95ptile_exp200km_pm1day_MERRA2.nc", \
               "geolines_PRECT_aint_95ptile_exp300km_pm1day_MERRA2.nc", \
               "geolines_PRECT_aint_95ptile_exp400km_pm1day_MERRA2.nc", \
               "geolines_PRECT_aint_95ptile_exp500km_pm1day_MERRA2.nc", \
               "geolines_PRECT_aint_95ptile_exp600km_pm1day_MERRA2.nc", \
               "geolines_PRECT_aint_95ptile_exp700km_pm1day_MERRA2.nc", \
               "geolines_PRECT_aint_95ptile_exp800km_pm1day_MERRA2.nc", \
               "geolines_PRECT_aint_95ptile_exp900km_pm1day_MERRA2.nc", \
               "geolines_PRECT_aint_95ptile_exp1000km_pm1day_MERRA2.nc", \
               "geolines_PRECT_aint_95ptile_exp1100km_pm1day_MERRA2.nc", \
               "geolines_PRECT_aint_95ptile_exp1200km_pm1day_MERRA2.nc"/)


labels = (/"f19", \
           "f09", \
           "ne30pg2", \
           "ne30pg3", \
           "ARCTIC", \
           "ARCTICGRIS", \
           "ERA5", \
           "MERRA2"/)

labels  = (/"LL_2~S~o~N~", \
            "LL_1~S~o~N~", \
            "QU_1.5~S~o~N~", \
            "QU_1~S~o~N~", \
            "VR_0.25~S~o~N~", \
            "VR_0.125~S~o~N~", \
            "ERA5", \
            "MERRA2"/)

  colors = (/"blue","deepskyblue","green4","green","darkorchid4","magenta","black","black","grey49"/)

  pltNam = "temp_expand"

;******************************************************
; create plot
;******************************************************
  wkstype = "pdf"
  wks = gsn_open_wks(wkstype,pltNam)               ; open workstation

  plot = new( (/4/), "graphic")

   res             = True
   res@gsnMaximize = False
   res@gsnDraw     = False      ; Will draw later, after overlaying
   res@gsnFrame    = False      ; all plots

   res@vpWidthF            = 0.7           ; change aspect ratio of plot
   res@vpHeightF           = 0.3

   res@tmXMajorGridThicknessF = 2.
   res@tmYMajorGridThicknessF = 2.
   res@tmXMinorGridThicknessF = 1.
   res@tmYMinorGridThicknessF = 1.

   res@tmXBMajorThicknessF = 2.
   res@tmYRMajorThicknessF = 2.
   res@tmXTMajorThicknessF = 2.
   res@tmYLMajorThicknessF = 2.

   res@tmBorderThicknessF = 3.

   res@tmXBMajorOutwardLengthF = 0.
   res@tmYLMajorOutwardLengthF = 0.
   res@tmXBMinorOutwardLengthF = 0.
   res@tmYLMinorOutwardLengthF = 0.
   res@tmXTMajorOutwardLengthF = 0.
   res@tmYRMajorOutwardLengthF = 0.
   res@tmXTMinorOutwardLengthF = 0.
   res@tmYRMinorOutwardLengthF = 0.

   res@tiXAxisFontHeightF = 0.017
   res@tiYAxisFontHeightF = 0.017
   res@tmXBLabelFontHeightF = 0.015
   res@tmYLLabelFontHeightF = 0.017

   ;;res@gsnXRefLine = 500.

   res@tmGridDrawOrder = "PreDraw"
   res@tmXMajorGrid= False
   res@tmXMajorGridLineColor = "grey40"
   res@tmXMajorGridThicknessF      = 1.
   ;;res@tmXMinorGrid= True
   res@tmXMinorGridLineColor = "grey40"
   res@tmXMinorGridThicknessF      = 1.
   res@tmXMinorGridLineDashPattern = 2
   res@tmYMajorGrid= False
   res@tmYMajorGridLineColor = "grey40"
   res@tmYMajorGridThicknessF      = 1.
   ;;res@tmYMinorGrid= True
   res@tmYMinorGridLineColor = "grey40"
   res@tmYMinorGridThicknessF      = 1.
   res@tmYMinorGridLineDashPattern = 2
   
   ;;; transparent polygons
   gsres                = True
   gsres@gsFillOpacityF = 0.10;;0.2;;0.3
   gsres@tfPolyDrawOrder = "PreDraw"
 

;******************************************************
; create plot
;******************************************************

  dtcam = 0.25

  yaxisa = "Precip Rate (mm)"
  ymin1 = 0.
  ymax1 = 75.
  ymin2 = 0.
  ymax2 = 190.

  yaxisb = "Area Int Precip (Gt)"
  ymin3 = 0.
  ymax3 = 9.
  ymin4 = 0.
  ymax4 = 25.

  xaxis = "Radial Great Circle Distance (km)"
  xmin = 0.
  xmax = 1200.

  rdist = fspan(0,1200.,ndist)

  lines1 = new( (/nmodel,ndist/), "double")
  ups1 = lines1
  downs1 = lines1
  lines2 = new( (/nmodel,ndist/), "double")
  ups2 = lines2
  downs2 = lines2
  lines3 = new( (/nmodel,ndist/), "double")
  ups3 = lines3
  downs3 = lines3
  lines4 = new( (/nmodel,ndist/), "double")
  ups4 = lines4
  downs4 = lines4

  lines1b = new( (/ndist/), "double")
  ups1b = lines1b
  downs1b = lines1b

;******************************************************
; begin loop
;******************************************************
     
  do n = 0,nmodel-1

   do m = 0,ndist-1

     print("load:"+subdir_scrip+fname0(m,n))
     f = addfile(subdir_scrip + fname0(m,n),"r")
     var1 = f->av
     xp1 = f->xp
     yp1 = f->yp
     delete(f)

     camcum = var1
     camcum = 0.
     n1 = dimsizes(camcum)

     up = yp1(1:n1)
     tmp= yp1(n1+1:2*n1)
     dn = tmp(::-1)
     delete(tmp)

     cumup = camcum
     cumup = 0.
     cumdn = cumup
     do q = 1,n1-1
       camcum(q) = camcum(q-1) + dtcam*var1(q)
       cumup(q) = cumup(q-1) + dtcam*up(q)
       cumdn(q) = cumdn(q-1) + dtcam*dn(q)
     end do
     delete([/up,dn/])
     lines1(n,m) = camcum(n1-1)
     ups1(n,m) = cumup(n1-1)
     downs1(n,m) = cumdn(n1-1)
     delete([/camcum,cumup,cumdn,var1/])

     if (n.eq.nxtra) then

       print("load:"+subdir_scrip+tname1(m))
       f = addfile(subdir_scrip + tname1(m),"r")
       var1b = f->av
       xp1b = f->xp
       yp1b = f->yp
       delete(f)

       camcum = var1b
       camcum = 0.
       n1b = dimsizes(camcum)

       up = yp1b(1:n1b)
       tmp= yp1b(n1b+1:2*n1b)
       dn = tmp(::-1)
       delete(tmp)

       cumup = camcum
       cumup = 0.
       cumdn = cumup
       do q = 1,n1b-1
         camcum(q) = camcum(q-1) + dtcam*var1b(q)
         cumup(q) = cumup(q-1) + dtcam*up(q)
         cumdn(q) = cumdn(q-1) + dtcam*dn(q)
       end do
       delete([/up,dn/])
       lines1b(m) = camcum(n1b-1)
       ups1b(m) = cumup(n1b-1)
       downs1b(m) = cumdn(n1b-1)
       delete([/camcum,cumup,cumdn,var1b/])

     end if

     print("load:"+subdir_scrip+fname1(m,n))
     f = addfile(subdir_scrip + fname1(m,n),"r")
     var2 = f->av
     xp2 = f->xp
     yp2 = f->yp
     delete(f)

     camcum = var2
     camcum = 0.
     n2 = dimsizes(camcum)

     up = yp2(1:n2)
     tmp= yp2(n2+1:2*n2)
     dn = tmp(::-1)
     delete(tmp)

     cumup = camcum
     cumup = 0.
     cumdn = cumup
     do q = 1,n2-1
       camcum(q) = camcum(q-1) + dtcam*var2(q)
       cumup(q) = cumup(q-1) + dtcam*up(q)
       cumdn(q) = cumdn(q-1) + dtcam*dn(q)
     end do
     delete([/up,dn/])
     lines2(n,m) = camcum(n2-1)
     ups2(n,m) = cumup(n2-1)
     downs2(n,m) = cumdn(n2-1)
     delete([/camcum,cumup,cumdn,var2/])

     print("load:"+subdir_scrip+fname2(m,n))
     f = addfile(subdir_scrip + fname2(m,n),"r")
     var3 = f->av
     xp3 = f->xp
     yp3 = f->yp
     delete(f)

     camcum = var3
     camcum = 0.
     n3 = dimsizes(camcum)

     up = yp3(1:n3)
     tmp= yp3(n3+1:2*n3)
     dn = tmp(::-1)
     delete(tmp)

     cumup = camcum
     cumup = 0.
     cumdn = cumup
     do q = 1,n3-1
       camcum(q) = camcum(q-1) + dtcam*var3(q)
       cumup(q) = cumup(q-1) + dtcam*up(q)
       cumdn(q) = cumdn(q-1) + dtcam*dn(q)
     end do
     delete([/up,dn/])
     lines3(n,m) = camcum(n3-1)
     ups3(n,m) = cumup(n3-1)
     downs3(n,m) = cumdn(n3-1)
     delete([/camcum,cumup,cumdn,var3/])

     print("load:"+subdir_scrip+fname3(m,n))
     f = addfile(subdir_scrip + fname3(m,n),"r")
     var4 = f->av
     xp4 = f->xp
     yp4 = f->yp
     delete(f)
   
     camcum = var4
     camcum = 0.
     n4 = dimsizes(camcum)
     
     up = yp4(1:n4)
     tmp= yp4(n4+1:2*n4)
     dn = tmp(::-1)
     delete(tmp)

     cumup = camcum
     cumup = 0.
     cumdn = cumup 
     do q = 1,n4-1
       camcum(q) = camcum(q-1) + dtcam*var4(q)
       cumup(q) = cumup(q-1) + dtcam*up(q)
       cumdn(q) = cumdn(q-1) + dtcam*dn(q)
     end do
     delete([/up,dn/])
     lines4(n,m) = camcum(n4-1)
     ups4(n,m) = cumup(n4-1)
     downs4(n,m) = cumdn(n4-1)
     delete([/camcum,cumup,cumdn,var4/])

   end do

   res@xyDashPatterns = 0
   if (n.eq.nmodel-1) then
     res@xyDashPatterns = 16
   end if
   res@xyLineThicknessF  = 4
   res@xyLineColors = colors(n)

   if (n.eq.0) then
     res@trXMinF         = xmin
     res@trXMaxF         = xmax

     res@tiYAxisString   = yaxisa
     res@trYMinF         = ymin1
     res@trYMaxF         = ymax1
     plot(0)  = gsn_csm_xy(wks,rdist,lines1(n,:),res)

     res@trYMinF         = ymin2
     res@trYMaxF         = ymax2
     plot(1)  = gsn_csm_xy(wks,rdist,lines2(n,:),res)

     res@tmGridDrawOrder = "PreDraw"
     res@tmXMajorGrid= True
     res@tmXMajorGridLineColor = "grey40"
     res@tmXMajorGridThicknessF      = 1.
     ;;res@tmXMinorGrid= True
     res@tmXMinorGridLineColor = "grey40"
     res@tmXMinorGridThicknessF      = 1.
     res@tmXMinorGridLineDashPattern = 2
     res@tmYMajorGrid= True
     res@tmYMajorGridLineColor = "grey40"
     res@tmYMajorGridThicknessF      = 1.
     ;;res@tmYMinorGrid= True
     res@tmYMinorGridLineColor = "grey40"
     res@tmYMinorGridThicknessF      = 1.
     res@tmYMinorGridLineDashPattern = 2

     res@tiXAxisString   = xaxis

     res@tiYAxisString   = yaxisb
     res@trYMinF         = ymin3
     res@trYMaxF         = ymax3
     plot(2)  = gsn_csm_xy(wks,rdist,lines3(n,:),res)

     res@trYMinF         = ymin4
     res@trYMaxF         = ymax4
     plot(3)  = gsn_csm_xy(wks,rdist,lines4(n,:),res)
   else
     tmp = gsn_csm_xy(wks,rdist,lines1(n,:),res)
     overlay(plot(0),tmp)
     delete(tmp)

     tmp = gsn_csm_xy(wks,rdist,lines2(n,:),res)
     overlay(plot(1),tmp)
     delete(tmp)

     tmp = gsn_csm_xy(wks,rdist,lines3(n,:),res)
     overlay(plot(2),tmp)
     delete(tmp)

     tmp = gsn_csm_xy(wks,rdist,lines4(n,:),res)
     overlay(plot(3),tmp)
     delete(tmp)
   end if

   if (n.eq.nxtra) then
     res@xyDashPatterns = 14;;2
     res@xyLineThicknessF = 4;;2
     tmp = gsn_csm_xy(wks,rdist,lines1b,res)
     overlay(plot(1),tmp)
     delete(tmp)
     res@xyLineThicknessF = 4
     res@xyDashPatterns = 0
   end if

   ;;now plot polygon shading
   gsres@gsFillColor    = colors(n)

   if (n.eq.0) then
     m1 = 2*ndist+1
     xxp1 = new( (/m1/), "float")
     xxp1(0) = rdist(0)
     xxp1(1:ndist) = rdist
     xxp1(ndist+1:2*ndist) = rdist(::-1) 
   end if

   yyp1 = new( (/m1/), "double")
   yyp1(0) = downs1(n,0)
   yyp1(1:ndist) = (/ups1(n,:)/)
   yyp1(ndist+1:2*ndist) = (/downs1(n,::-1)/)

   yyp2 = new( (/m1/), "double")
   yyp2(0) = downs2(n,0)
   yyp2(1:ndist) = (/ups2(n,:)/)
   yyp2(ndist+1:2*ndist) = (/downs2(n,::-1)/)

   yyp3 = new( (/m1/), "double")
   yyp3(0) = downs3(n,0) 
   yyp3(1:ndist) = (/ups3(n,:)/)
   yyp3(ndist+1:2*ndist) = (/downs3(n,::-1)/)

   yyp4 = new( (/m1/), "double")
   yyp4(0) = downs4(n,0)
   yyp4(1:ndist) = (/ups4(n,:)/)
   yyp4(ndist+1:2*ndist) = (/downs4(n,::-1)/)

   polystr = unique_string("hum")
   plot@$polystr$ = gsn_add_polygon(wks,plot(0),xxp1,yyp1,gsres)
   delete(yyp1)

   polystr = unique_string("hum")
   plot@$polystr$ = gsn_add_polygon(wks,plot(1),xxp1,yyp2,gsres)
   delete(yyp2)

   polystr = unique_string("hum")
   plot@$polystr$ = gsn_add_polygon(wks,plot(2),xxp1,yyp3,gsres)
   delete(yyp3)

   polystr = unique_string("hum")
   plot@$polystr$ = gsn_add_polygon(wks,plot(3),xxp1,yyp4,gsres)
   delete(yyp4)

;;too busy
;   if (n.eq.nxtra) then
;     yyp1b = new( (/m1/), "double")
;     yyp1b(0) = downs1b(0)
;     yyp1b(1:ndist) = (/ups1b/)
;     yyp1b(ndist+1:2*ndist) = (/downs1b(::-1)/)
;
;     polystr = unique_string("hum")
;     plot@$polystr$ = gsn_add_polygon(wks,plot(1),xxp1,yyp1b,gsres)
;     delete(yyp1b)
;   end if

 end do ;;n
    
;*****************************************************
; Add legend
;*****************************************************
 txres = True
 txres@txFontHeightF = 0.010
 txres@txJust = "CenterLeft"

 resln             = True
 resln@gsLineThicknessF = res@xyLineThicknessF

 yy = (/0.74,0.725,0.71,0.695,0.68,0.665,0.65,0.635/)-0.075;;-0.07
 xx = 0.82;;0.8
 dx = 0.03
 do n = 0,nmodel-1
   resln@gsLineColor = colors(n)
   resln@gsLineDashPattern = 0
   if (n.eq.nmodel-1) then
     resln@gsLineDashPattern = 16
   end if
   gsn_polyline_ndc(wks,(/xx,xx+dx/),(/yy(n),yy(n)/),resln)
   gsn_text_ndc(wks,"  " + labels(n),xx+dx,yy(n),txres)
 end do

 txres@txFontHeightF = 0.015
 txres@txJust = "CenterCenter"

 delete(yy)
 yy = 0.70
 xx = 0.30
 gsn_text_ndc(wks,"Mean AR",xx,yy,txres)

 xx = 0.75
 gsn_text_ndc(wks,"95th Percentile AR",xx,yy,txres)

;;drawNDCGrid(wks)
;*****************************************************
; Plot panel
;*****************************************************

  resP                  = True                   ; modify the panel plot
  resP@gsnFrame         = False                  ; don't advance panel plot

  ;;resP@txString = plotTitle
  resP@gsnPanelFigureStrings= (/"a.","b.","c.","d."/)
  resP@gsnPanelFigureStringsFontHeightF = 0.012
  resP@gsnPanelFigureStringsPerimOn     = True
  resP@gsnPanelFigureStringsBackgroundFillColor = "white"
  resP@amJust           = "BottomRight"
  resP@gsnPanelLeft = 0.05
  resP@gsnPanelRight = 0.95
  ;;resP@gsnPanelBottom   = 0.1

  gsn_panel(wks,plot,(/2,2/),resP)

frame(wks)

end
