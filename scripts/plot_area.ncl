;**************************************
;var_cam.ncl
;**************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"
load "/glade/work/aherring/CESM2/arh-git-scripts/ncl/se_func_lib.ncl"
;**************************************
onemnth = False

;;list of cmasks

dir = (/"/glade/work/awaling/AR_GrIS/ESMF/f19/f19_GrIS_blobs/", \
            "/glade/work/awaling/AR_GrIS/ESMF/ne30pg2/ne30pg2_GrIS_blobs/", \
            "/glade/work/awaling/AR_GrIS/TempestRemap/f19/f19_GrIS_blobs/", \
            "/glade/work/awaling/AR_GrIS/TempestRemap/ne30pg2/ne30pg2_GrIS_blobs/"/)

dir_scrip = (/"../data/scrip/ESMF/f19/", \
              "../data/scrip/ESMF/ne30pg2/", \ 
              "../data/scrip/TempestRemap/f19/", \
              "../data/scrip/TempestRemap/ne30pg2/"/)

dir_s = dimsizes(dir_scrip)

subdir = (/"cam6_2_022.se_FHIST_f19_f19_mg17_900pes_200506_mg3-Nx5yrs/", \
            "cam6_2_022.se_FHIST_f09_f09_mg17_1800pes_200507_mg3-Nx5yrs/", \
            "cam6_2_022.se_FHIST_ne30pg2_ne30pg2_mg17_1800pes_200507_mg3-Nx5yrs/", \
            "cam6_2_022.se_FHIST_ne30pg3_ne30pg3_mg17_1800pes_200507_mg3-Nx5yrs/", \
            "cam6_2_022.se_FHIST_ne0np4.ARCTIC.ne30x4_mt12_7680pes_200507_mg3-Nx2yrs/", \
            "cam6_2_022.se_FHIST_ne0np4.ARCTICGRIS.ne30x8_mt12_7680pes_200510_mg3-Nx1yrs/", \
            "ERA5/", \
            "MERRA2/"/)

subdir_s = dimsizes(subdir)

grid_res = (/"f19", \
             "f09", \
             "ne30pg2", \
             "ne30pg3", \
             "ARCTIC", \
             "ARCTICGRIS", \
             "ERA5", \
             "MERRA2" /)

grid_res  = (/"LL_2~S~o~N~", \
            "LL_1~S~o~N~", \
            "QU_1.5~S~o~N~", \
            "QU_1~S~o~N~", \
            "VR_0.25~S~o~N~", \
            "VR_0.125~S~o~N~", \
            "ERA5", \
            "MERRA2"/)

file_n = "*cmasks.nc"

varsum_f19_2_f19 = new((/96,144/),integer)
varsum_f19_2_f19 = 0
varsum_f19_2_pg2 = new((/21600/),integer)
varsum_f19_2_pg2 = 0
varsum_f09_2_f19 = new((/96,144/),integer)
varsum_f09_2_f19 = 0
varsum_f09_2_pg2 = new((/21600/),integer)
varsum_f09_2_pg2 = 0
varsum_ne30pg2_2_f19 = new((/96,144/),integer)
varsum_ne30pg2_2_f19 = 0
varsum_ne30pg2_2_pg2 = new((/21600/),integer)
varsum_ne30pg2_2_pg2 = 0
varsum_ne30pg3_2_f19 = new((/96,144/),integer)
varsum_ne30pg3_2_f19 = 0
varsum_ne30pg3_2_pg2 = new((/21600/),integer)
varsum_ne30pg3_2_pg2 = 0
varsum_ARCTIC_2_f19 = new((/96,144/),integer)
varsum_ARCTIC_2_f19 = 0
varsum_ARCTIC_2_pg2 = new((/21600/),integer)
varsum_ARCTIC_2_pg2 = 0
varsum_ARCTICGRIS_2_f19 = new((/96,144/),integer)
varsum_ARCTICGRIS_2_f19 = 0
varsum_ARCTICGRIS_2_pg2 = new((/21600/),integer)
varsum_ARCTICGRIS_2_pg2 = 0
varsum_ERA5_2_f19 = new((/96,144/),integer)
varsum_ERA5_2_f19 = 0
varsum_ERA5_2_pg2 = new((/21600/),integer)
varsum_ERA5_2_pg2 = 0
varsum_MERRA2_2_f19 = new((/96,144/),integer)
varsum_MERRA2_2_f19 = 0
varsum_MERRA2_2_pg2 = new((/21600/),integer)
varsum_MERRA2_2_pg2 = 0

;;;GLACIER_REGION = 2, GREENLAND
rid = 2

;;;PCT_LANDUNIT =3, GLACIER
gid = 3

pltNam = "temp_area"

cellfill = True
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;*****************************************************
; create plot
;*****************************************************
  wkstype = "pdf"
  wks = gsn_open_wks(wkstype,pltNam)               ; open workstation
  ;;gsn_define_colormap(wks,"cmp_b2r")
  gsn_define_colormap(wks,"BlGrYeOrReVi200")
  ;gsn_reverse_colormap(wks)
  plot = new( (/16/), "graphic")

   res             = True
   res@gsnMaximize = False
   res@gsnDraw     = False      ; Will draw later, after overlaying
   res@gsnFrame    = False      ; all plots

   res@cnFillOn          = True               ; turn on color fill
   res@cnLinesOn         = False
   res@cnLineLabelsOn    = False
   
   ;res@cnLevelSelectionMode = True
   ;res@cnMaxLevelValF       = 3
   ;res@cnMinLevelValF       = 0
   ;res@cnLevelSpacingF      = 1   

   res@cnLevelSelectionMode = "ExplicitLevels"
   ;res@cnLevels = (/1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,20.,40.,60.,80.,100.,125.,150.,175.,200.,225.,250.,275.,300.,325.,350.,375.,400.,425.,450.,475.,500./)
   ;res@cnLevels = (/1.,2.,3.,4.,5.,25.,50.,75.,100.,125.,150.,175.,200.,225.,250.,275.,300.,325.,350.,375.,400.,425.,450.,475.,500./)    
   res@cnLevels = (/1.,2.,3.,4.,5.,10.,50.,100.,200.,300.,400.,500./)

   res@cnConstFEnableFill = True

   res@gsnSpreadColors   = True               ; use full range of color map
   res@gsnSpreadColorStart = 5
   res@cnInfoLabelOn        = False
   res@lbLabelBarOn         = False
   res@tiMainFontHeightF = 0.006

   res@vpWidthF  = 0.3
   res@vpHeightF = 0.8

   res@mpGridAndLimbOn   = False           ; turn on lat/lon lines
   res@mpGridLatSpacingF = 10.
   res@mpGridLonSpacingF = 30.
   ;res@mpGridLineColor      = -1
   res@mpPerimOn         = True               ; turn off perimeter
   res@mpFillOn         = True
   ;;res@mpOutlineBoundarySets     = "NoBoundaries"

   res@mpProjection = "LambertConformal";;"Orthographic";;"Mollweide"
   res@mpLambertParallel1F = 85
   res@mpLambertParallel2F = 58
   res@mpLambertMeridianF  = -45
   res@mpLimitMode       = "LatLon"
   res@mpMinLonF = -56.4
   res@mpMaxLonF = -30.
   res@mpMinLatF = 59.4;;58.
   res@mpMaxLatF = 85;;85.

;*****************************************************

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

i=0

do n = 0,dir_s-1

   do m = 0,subdir_s-1
        fname = systemfunc ("ls " +  dir_scrip(n) + subdir(m) + file_n) 
	nhist = dimsizes(fname)
        ;print(""+fname)

        f  = addfile (fname(0),"r")
        dims = getfiledimsizes(f)
        dimnames = getfiledimnames(f)
        var = f->cmasks
        delete(f)

        ;var = ind(dimnames.eq."cmasks")
        ;print(var)
 
        if (any(dimnames.eq."ncl2")) then
          latlon = True
        else
          latlon = False
        end if

    	;;set all missing values to zero
	var@_FillValue = default_fillvalue("integer")
	var = where(ismissing(var),0,var) 
;print("max cmasks " + max(var))
;print("min cmasks " + min(var))
       
	;;sum over the nevents index <---this will give you a number of times a storm occupied a particular grid point. This is what we want to plot. It's similar to an AR frequency plot. 
        ;printVarSummary(var)
	varsum = dim_sum_n(var,0)
;print("max varsum " + max(varsum))
;print("min varsum " + min(varsum))
	;plot varsum!
	;now to get the combined area occupied by all blobs for each run (this will be for the table)

	;;set everything >1 in varsum, to 1.
	varone = where(varsum.gt.1,1,varsum)

        if (latlon) then
         ;print("get grid areas")
         ilat = ind(dimnames.eq."ncl1")
         ilon = ind(dimnames.eq."ncl2")
         domains = get_fv_domain(dims(ilat),dims(ilon))
         area = domains[0]
         lmask= domains[1]
         lattmp = domains[2]
         lontmp = domains[3]
         ;print(lattmp)
	 ;print(lontmp)
  	 lat = lattmp(:)
  	 lon = lontmp(:)
  	 ;print(lat)
 	 ;print(lon)

         delete(domains)
         delete([/ilat,ilon/])

         ;print("get clmout")
         clmdir = "/glade/campaign/cgd/amp/aherring/archive/"
         clmdir_scrip = "../data/scrip/"
         clmout = "cam6_2_022.se_FHIST_f19_f19_mg17_900pes_200506_mg3-Nx5yrs"

         cfiles = systemfunc ("ls " + clmdir_scrip + clmout + "/lnd/hist/" + clmout + ".clm2.h0.*")
         cfiles_scrip = systemfunc ("ls " + clmdir_scrip + clmout + ".clm2.h0.*")
         sf = addfile (cfiles_scrip(0),"r")
         pctglc = sf->PCT_LANDUNIT(0,gid,:,:)
         delete([/cfiles,sf/])

         surfdir= "/glade/work/aherring/grids/uniform-res/f19/clm_surfdata_5_0/"
         surfdir_scrip = "../data/scrip/"
         surfdat= "surfdata_f19_hist_78pfts_CMIP6_simyr1850_c200429.nc"
       else
         ;print("get grid areas")
         icol = ind(dimnames.eq."ncl1")
         domains = get_se_domain(dims(icol))
         ;print(domains)
         lattmp = domains[2]
         lontmp = domains[3]
         ;domains = get_se_domain(dims(0))
         tmp0 = domains[0]
         area = tmp0(0,:)
         tmp1 = domains[1]
         lmask = tmp1(0,:)
	 lat = lattmp(:)
         lon = lontmp(:)
         ;delete([/tmp0,tmp1,icol,domains/])

         ;print("get clmout")
         clmdir = "/glade/campaign/cgd/amp/aherring/archive/"
         clmdir_scrip = "../data/scrip/"
         clmout = "cam6_2_022.se_FHIST_ne30pg2_ne30pg2_mg17_1800pes_200507_mg3-Nx5yrs"

         cfiles = systemfunc ("ls " + clmdir_scrip + clmout + "/lnd/hist/" + clmout + ".clm2.h0.*")
         cfiles_scrip = systemfunc ("ls " + clmdir_scrip + clmout + ".clm2.h0.*")
         sf = addfile (cfiles_scrip(0),"r")
         pctglc = sf->PCT_LANDUNIT(0,gid,:)
         delete([/cfiles,sf/])
   
         surfdir= "/glade/work/aherring/grids/uniform-res/ne30np4.pg2/clm_surfdata_5_0/"
         surfdir_scrip = "../data/scrip/"
         surfdat= "surfdata_ne30np4.pg2_hist_78pfts_CMIP6_simyr1850_c200426.nc"
   end if
     delete([/lattmp,lontmp/])
     ;print("get surfdata")
     sf = addfile(surfdir_scrip+surfdat,"r")
     regionmask = sf->GLACIER_REGION
     regionmask@_FillValue = default_fillvalue("integer")
     regionmask = where(regionmask.eq.rid,1,regionmask@_FillValue)
     delete(sf)

     pctglc = pctglc/100.
     ;printVarSummary(area)
     ;printVarSummary(Re)
     ;printVarSummary(lmask)
     ;printVarSummary(pctglc)
     ;printVarSummary(regionmask)
     area = Re^2.*area*lmask*todouble(pctglc)*todouble(regionmask)
     delete([/lmask,pctglc,regionmask/])
     
     area@FillValue = default_fillvalue("float")
     area = where(ismissing(area),0,area)

if (latlon) then
  totarea = dim_sum(dim_sum(area*varone))
  ;print(varsum)
  ;print(area)
  ; print name of file and totarea
  print(fname)
  print(totarea)
else
  totarea = dim_sum(area*varone)
  ; print name of file and totarea
  print(fname)
  print(totarea)
end if

;;set the zero's back to missing values <---for plotting aesctics
varsum@_FillValue = default_fillvalue("integer")
varsum = where(varsum.eq.0,varsum@_FillValue,varsum)

plotTitle = "temp_area"

   if (.not.latlon) then
     res@sfXArray = lon
     res@sfYArray = lat
   end if

   if (cellfill) then
     if (latlon) then
       ;;;Grid descriptor
       griddir = (/"/glade/work/aherring/grids/uniform-res/f19/grids/"/)
       griddir_scrip = "../data/scrip/"
       scrip = (/"fv1.9x2.5_141008.nc"/)
     else
       griddir = (/"/glade/work/aherring/grids/uniform-res/ne30np4.pg2/grids/"/)
       griddir_scrip = "../data/scrip/"
       scrip = (/"ne30pg2_scrip_170608.nc"/)
     end if

     res@cnFillMode = "CellFill"
     vf = addfile(griddir_scrip+scrip,"r")
     res@sfXCellBounds = vf->grid_corner_lon
     res@sfYCellBounds = vf->grid_corner_lat
     delete(vf)
   else
     res@cnFillMode = "AreaFill"
   end if

if (latlon) then
       ;printVarSummary(varsum)
       ;varsump = dim_sum_n(varsum,0)
       ;printVarSummary(varsump)
       ;varsum!0 = "lat"
       ;varsum&lat = lat
       ;varsum&lat@units = "degrees_north"
       ;varsum!1 = "lon"
       ;varsum&lon = lon
       ;varsum&lon@units = "degrees_east"
     else
       ;printVarSummary(varsum)
       ;varsump = dim_sum_n(varsum,0)
end if

if (isStrSubset(subdir(m),"f19")) then
;if (subdir(m).eq."cam6_2_022.se_FHIST_f19_f19_mg17_900pes_200506_mg3-Nx5yrs/") then
  if (latlon) then
    copy_VarCoords(varsum,varsum_f19_2_f19)
    varsum_f19_2_f19 = varsum + varsum_f19_2_f19
    varsum_f19_2_f19!0 = "lat"
    varsum_f19_2_f19&lat = lat
    varsum_f19_2_f19&lat@units = "degrees_north"
    varsum_f19_2_f19!1 = "lon"
    varsum_f19_2_f19&lon = lon
    varsum_f19_2_f19&lon@units = "degrees_east"
  else
    varsum_f19_2_pg2 = varsum_f19_2_pg2 + varsum
  end if   
else if (isStrSubset(subdir(m),"f09")) then
  if (latlon) then
  varsum_f09_2_f19 = varsum + varsum_f09_2_f19
  varsum_f09_2_f19!0 = "lat"
  varsum_f09_2_f19&lat = lat
  varsum_f09_2_f19&lat@units = "degrees_north"
  varsum_f09_2_f19!1 = "lon"
  varsum_f09_2_f19&lon = lon
  varsum_f09_2_f19&lon@units = "degrees_east"
  else
    varsum_f09_2_pg2 = varsum_f09_2_pg2 + varsum
  end if
else if (isStrSubset(subdir(m),"ne30pg2")) then
  if (latlon) then
  varsum_ne30pg2_2_f19 = varsum + varsum_ne30pg2_2_f19
  varsum_ne30pg2_2_f19!0 = "lat"
  varsum_ne30pg2_2_f19&lat = lat
  varsum_ne30pg2_2_f19&lat@units = "degrees_north"
  varsum_ne30pg2_2_f19!1 = "lon"
  varsum_ne30pg2_2_f19&lon = lon
  varsum_ne30pg2_2_f19&lon@units = "degrees_east"
  else
    varsum_ne30pg2_2_pg2 = varsum_ne30pg2_2_pg2 + varsum
  end if
else if (isStrSubset(subdir(m),"ne30pg3")) then
 if (latlon) then
 varsum_ne30pg3_2_f19 = varsum + varsum_ne30pg3_2_f19
  varsum_ne30pg3_2_f19!0 = "lat"
  varsum_ne30pg3_2_f19&lat = lat
  varsum_ne30pg3_2_f19&lat@units = "degrees_north"
  varsum_ne30pg3_2_f19!1 = "lon"
  varsum_ne30pg3_2_f19&lon = lon
  varsum_ne30pg3_2_f19&lon@units = "degrees_east" 
  else
    varsum_ne30pg3_2_pg2 = varsum_ne30pg3_2_pg2 + varsum
  end if
else if (isStrSubset(subdir(m),"ne30x4")) then ; this is a loop for ARCTIC
 if (latlon) then
 varsum_ARCTIC_2_f19 = varsum + varsum_ARCTIC_2_f19
  varsum_ARCTIC_2_f19!0 = "lat"
  varsum_ARCTIC_2_f19&lat = lat
  varsum_ARCTIC_2_f19&lat@units = "degrees_north"
  varsum_ARCTIC_2_f19!1 = "lon"
  varsum_ARCTIC_2_f19&lon = lon
  varsum_ARCTIC_2_f19&lon@units = "degrees_east"
  else
    varsum_ARCTIC_2_pg2 = varsum_ARCTIC_2_pg2 + varsum
  end if
else if (isStrSubset(subdir(m),"GRIS")) then 
if (latlon) then
 varsum_ARCTICGRIS_2_f19 = varsum + varsum_ARCTICGRIS_2_f19
  varsum_ARCTICGRIS_2_f19!0 = "lat"
  varsum_ARCTICGRIS_2_f19&lat = lat
  varsum_ARCTICGRIS_2_f19&lat@units = "degrees_north"
  varsum_ARCTICGRIS_2_f19!1 = "lon"
  varsum_ARCTICGRIS_2_f19&lon = lon
  varsum_ARCTICGRIS_2_f19&lon@units = "degrees_east"
  else 
    varsum_ARCTICGRIS_2_pg2 = varsum_ARCTICGRIS_2_pg2 + varsum
  end if
else if (isStrSubset(subdir(m),"ERA")) then
  if (latlon) then
   varsum_ERA5_2_f19 = varsum + varsum_ERA5_2_f19
   varsum_ERA5_2_f19!0 = "lat"
   varsum_ERA5_2_f19&lat = lat
   varsum_ERA5_2_f19&lat@units = "degrees_north"
   varsum_ERA5_2_f19!1 = "lon"
   varsum_ERA5_2_f19&lon = lon
   varsum_ERA5_2_f19&lon@units = "degrees_east"
  else 
   varsum_ERA5_2_pg2 = varsum_ERA5_2_pg2 + varsum   
   end if
else if (isStrSubset(subdir(m),"MERRA2")) then
  if (latlon) then
   varsum_MERRA2_2_f19 = varsum + varsum_MERRA2_2_f19
   varsum_MERRA2_2_f19!0 = "lat"
   varsum_MERRA2_2_f19&lat = lat
   varsum_MERRA2_2_f19&lat@units = "degrees_north"
   varsum_MERRA2_2_f19!1 = "lon"
   varsum_MERRA2_2_f19&lon = lon
   varsum_MERRA2_2_f19&lon@units = "degrees_east"
  else
   varsum_MERRA2_2_pg2 = varsum_MERRA2_2_pg2 + varsum
if (n.eq.3) then
       print("Made it to last file")
       plot_lat = new((/21600/),double)
       plot_lon = new((/21600/),double)
       plot_lat = lat
       plot_lon = lon
       plot_XCell = res@sfXCellBounds
       plot_YCell = res@sfYCellBounds
    end if
  end if
end if
end if
end if
end if
end if
end if
end if
end if
   if (.not.latlon) then
     delete([/res@sfXArray,res@sfYArray/])
   end if

   if (cellfill) then
     delete([/res@sfXCellBounds,res@sfYCellBounds/])
   end if


if (dir(n).eq."/glade/work/awaling/AR_GrIS/ESMF/f19/f19_GrIS_blobs/") then
  delete([/res@gsnCenterString/])
end if

;print("Is varsum all missing ? " + all(ismissing(varsum))) 

delete([/var,varsum,varone,dims,dimnames,area,lat,lon/])

i = i+1

end do
end do

varsum_f19_2_f19 = varsum_f19_2_f19/2
varsum_f09_2_f19 = varsum_f09_2_f19/2
varsum_ne30pg2_2_f19 = varsum_ne30pg2_2_f19/2
varsum_ne30pg3_2_f19 = varsum_ne30pg3_2_f19/2
varsum_ARCTIC_2_f19 = varsum_ARCTIC_2_f19/2
varsum_ARCTICGRIS_2_f19 = varsum_ARCTICGRIS_2_f19/2
varsum_ERA5_2_f19 = varsum_ERA5_2_f19/2
varsum_MERRA2_2_f19 = varsum_MERRA2_2_f19;/2
;print("Is varsumf19 all missing ? " + all(ismissing(varsum_f19))) 

res@gsnCenterStringFontHeightF = 0.04

res@tiYAxisFontHeightF    = 0.04
res@tiYAxisString        = "LL_2~S~o~N~ remapped"
;res@tiYAxisJust          = "Center"
res@tiYAxisSide          = "Left"
res@tiYAxisPosition      = "Center"

res@gsnCenterString = "LL_2~S~o~N~"
plot(0) = gsn_csm_contour_map(wks,varsum_f19_2_f19,res)

res@tiYAxisString        = ""

res@sfXCellBounds        = plot_XCell
res@sfYCellBounds        = plot_YCell

res@gsnCenterString = "LL_1~S~o~N~"
plot(1) = gsn_csm_contour_map(wks,varsum_f09_2_f19,res)
res@gsnCenterString = "QU_1.5~S~o~N~"
plot(2) = gsn_csm_contour_map(wks,varsum_ne30pg2_2_f19,res)
res@gsnCenterString = "QU_1~S~o~N~"
plot(3) = gsn_csm_contour_map(wks,varsum_ne30pg3_2_f19,res)
res@gsnCenterString = "VR_0.25~S~o~N~"
plot(4) = gsn_csm_contour_map(wks,varsum_ARCTIC_2_f19,res)
res@gsnCenterString = "VR_0.125~S~o~N~"
plot(5) = gsn_csm_contour_map(wks,varsum_ARCTICGRIS_2_f19,res)
res@gsnCenterString = "ERA5"
;print(varsum_f09_2_f19)
;print(varsum_ERA5_2_f19)
;print(varsum_ERA5_2_pg2)
plot(6) = gsn_csm_contour_map(wks,varsum_ERA5_2_f19,res)
res@gsnCenterString = "MERRA2"
plot(7) = gsn_csm_contour_map(wks,varsum_MERRA2_2_f19,res)
;define lat lon for these guys

res@sfXArray = plot_lon
res@sfYArray = plot_lat

res@tiYAxisString        = "QU_1.5~S~o~N~ remapped"



res@gsnCenterString = ""
plot(8) = gsn_csm_contour_map(wks,varsum_f19_2_pg2,res)

res@tiYAxisString  = ""

plot(9) = gsn_csm_contour_map(wks,varsum_f09_2_pg2,res)
;res@gsnCenterString = "ne30pg2"
plot(10) = gsn_csm_contour_map(wks,varsum_ne30pg2_2_pg2,res)
;res@gsnCenterString = "ne30pg3"
plot(11) = gsn_csm_contour_map(wks,varsum_ne30pg3_2_pg2,res)
;res@gsnCenterString = "ARCTIC"
plot(12) = gsn_csm_contour_map(wks,varsum_ARCTIC_2_pg2,res)
;res@gsnCenterString = "ARCTICGRIS"
plot(13) = gsn_csm_contour_map(wks,varsum_ARCTICGRIS_2_pg2,res)
plot(14) = gsn_csm_contour_map(wks,varsum_ERA5_2_pg2,res)
plot(15) = gsn_csm_contour_map(wks,varsum_MERRA2_2_pg2,res)

;;;drawNDCGrid(wks)
;*****************************************************
; Plot panel
;*****************************************************

  resP                  = True                   ; modify the panel plot
  resP@gsnFrame         = False                  ; don't advance panel plot
  ;resP@gsnPaperOrientation = "portrait"

  ;resP@gsnPanelFigureStrings = runnams
  resP@gsnPanelFigureStringsFontHeightF = 0.015
  resP@gsnPanelFigureStringsPerimOn     = True
  resP@gsnPanelFigureStringsBackgroundFillColor = "white"
  resP@amJust           = "TopLeft"

  ;resP@txString = plotTitle

  resP@gsnPanelLabelBar = True
  resP@lbBoxSeparatorLinesOn= False
  resP@gsnPanelBottom   = 0.1
  resP@lbTitleString              = "Number of times ARs intersect gridcell"
  ;resP@lbTitleString              = lbtitle
  resP@lbTitlePosition            = "Bottom"
  resP@lbTitleFontHeightF         = 0.015
  resP@lbLabelFontHeightF         = 0.012
  resP@pmLabelBarOrthogonalPosF   = -0.02

  res@vpWidthF  = 0.3		; set width of plot 
  res@vpHeightF = 0.8		; set height of plot

  ;resP@txString = pltTitle
  gsn_panel(wks,plot,(/2,8/),resP)

frame(wks)
exit
;**********************************************************************
; Add rotated text strings on left side of paneled plots.
; Use the resized plots to determine the position of the titles
;**********************************************************************
    pos1 = get_title_position(plot(0))         ; 1st row leftmost plot
    pos2 = get_title_position(plot(8))         ; 2nd row leftmost plot
    txres               = True
    txres@txFontHeightF = 0.012
    txres@txAngleF      = 90.
    gsn_text_ndc(wks, "f19 remapped",   pos1(0), pos1(1), txres)
    gsn_text_ndc(wks, "ne30pg2 remapped",   pos2(0), pos2(1), txres)
    ;gsn_text_ndc(wks, "f19 (TempestRemap)", pos3(0), pos3(1), txres)
    ;gsn_text_ndc(wks, "ne30pg2 (TempestRemap)",    pos4(0), pos4(1), txres)

frame(wks)


