;**************************************
;var_cam.ncl
;**************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"
load "/gpfs/fs1/work/aherring/CESM2/arh-git-scripts/ncl/se_func_lib.ncl"
;**************************************
begin
;****************************************
  p0 = 1000.d2
  secpday = 60.*60.*24;
  pi = 3.14159265358979
  g = 9.80616
  Re = 6371.22
  cp = 1004.
  Rd = 287.
;****************************************

  kframe = 4
  lbtitle = "PRECT (mm/day) LAG = +0 hrs"

  subdir_scrip = "../data/precip/"

  fname0 = (/"geolines_PRECT_amean_95ptile_f19.nc", \
            "geolines_PRECT_amean_95ptile_f09.nc", \
            "geolines_PRECT_amean_95ptile_ne30pg2.nc", \
            "geolines_PRECT_amean_95ptile_ne30pg3.nc", \
            "geolines_PRECT_amean_95ptile_ARCTIC.nc", \
            "geolines_PRECT_amean_95ptile_ARCTICGRIS.nc", \
            "geolines_PRECT_amean_95ptile_ERA5.nc", \
            "geolines_PRECT_amean_95ptile_MERRA2.nc"/)

  dir = "/glade/campaign/cgd/amp/aherring/archive/"
  rdir = "/atm/hist/"
  fname = (/"cam6_2_022.se_FHIST_f19_f19_mg17_900pes_200506_mg3-Nx5yrs", \
	    "cam6_2_022.se_FHIST_f09_f09_mg17_1800pes_200507_mg3-Nx5yrs", \
	    "cam6_2_022.se_FHIST_ne30pg2_ne30pg2_mg17_1800pes_200507_mg3-Nx5yrs", \
	    "cam6_2_022.se_FHIST_ne30pg3_ne30pg3_mg17_1800pes_200507_mg3-Nx5yrs", \
	    "cam6_2_022.se_FHIST_ne0np4.ARCTIC.ne30x4_mt12_7680pes_200507_mg3-Nx2yrs", \
	    "cam6_2_022.se_FHIST_ne0np4.ARCTICGRIS.ne30x8_mt12_7680pes_200510_mg3-Nx1yrs", \
	    "ERA5", \
	    "MERRA2"/)
  fdate = ".cam.h3.*"

  nera5 = 6
  era5dir = "/glade/campaign/cgd/amp/aherring/annelise_backup/TempestExtremes/ERA5_raw/"
  nmerra2 = 7
  merra2dir = "/glade/campaign/cgd/amp/aherring/annelise_backup/TempestExtremes/merra2_precip/"

  runnams = (/"f19", \
              "f09", \
              "ne30pg2", \
              "ne30pg3", \
              "ARCTIC", \
              "ARCTICGRIS", \
	      "ERA5", \
	      "MERRA2"/)

  nfiles = dimsizes(fname)

  VAR = "PRECT"
  factor = 1000.*secpday
  era5factor = 1000.*secpday/3600
  merra2factor = secpday

  labels = new( (/4*nfiles/), "string") 

  pltNam = "temp_contours"

  cellfill = True

  griddir = (/"/glade/work/aherring/grids/uniform-res/f19/grids/", \
              "/glade/work/aherring/grids/uniform-res/f09/grids/", \
              "/glade/work/aherring/grids/uniform-res/ne30np4.pg2/grids/", \
              "/glade/work/aherring/grids/uniform-res/ne30np4.pg3/grids/", \
              "/glade/work/aherring/grids/var-res/ne0np4.ARCTIC.ne30x4/grids/", \
              "/glade/work/aherring/grids/var-res/ne0np4.ARCTICGRIS.ne30x8/grids/", \
	      "/glade/work/aherring/grids/SCRIP_files/", \
              "/glade/work/aherring/grids/SCRIP_files/"/)

  scrip = (/"fv1.9x2.5_141008.nc", \
            "fv0.9x1.25_scrip_141008.nc", \
            "ne30pg2_scrip_170608.nc", \
            "ne30pg3_scrip_170611.nc", \
            "ne0ARCTICne30x4_scrip_191212.nc", \
            "ne0ARCTICGRISne30x8_scrip_191209.nc", \
	    "ERA5_025_221025.nc", \
	    "MERRA2_scrip.nc"/)

;******************************************************
; load dates?
;******************************************************

  tmppath = "/glade/campaign/cgd/amp/aherring/archive/cam6_2_022.se_FHIST_f19_f19_mg17_900pes_200506_mg3-Nx5yrs/atm/hist/cam6_2_022.se_FHIST_f19_f19_mg17_900pes_200506_mg3-Nx5yrs.cam.h3.*"
   
  sfiles = systemfunc ("ls " + tmppath)
  f = addfiles (sfiles,"r")
  ListSetType(f,"cat")
  delete(sfiles)

  dates = f[:]->date
  delete(f)

;******************************************************
; create plot
;******************************************************
  wkstype = "pdf"
  wks = gsn_open_wks(wkstype,pltNam)               ; open workstation

  gsn_define_colormap(wks,"WhiteBlueGreenYellowRed")
  ;;gsn_define_colormap(wks,"BkBlAqGrYeOrReViWh200")
  ;;gsn_define_colormap(wks,"cmp_b2r")
  ;;gsn_reverse_colormap(wks)

  plot = new( (/4*nfiles/), "graphic")

   res             = True
   res@gsnMaximize = False
   res@gsnDraw     = False      ; Will draw later, after overlaying
   res@gsnFrame    = False      ; all plots

   res@cnFillOn          = True               ; turn on color fill
   res@cnLinesOn         = False
   res@cnLineLabelsOn    = False

   res@gsnSpreadColors   = True               ; use full range of color map
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

;   res@mpProjection = "Orthographic"
;   res@mpCenterLatF      =   45.
;   res@mpCenterLonF      =  -45.
;
;   res@mpProjection = "LambertConformal";;"Orthographic";;"Mollweide"
;   res@mpLambertParallel1F = 85
;   res@mpLambertParallel2F = 58
;   res@mpLambertMeridianF  = -45
;   res@mpLimitMode       = "LatLon"
;   res@mpMinLonF = -56.4
;   res@mpMaxLonF = -30.
;   res@mpMinLatF = 59.4
;   res@mpMaxLatF = 85

   res@mpProjection = "LambertConformal"
   res@gsnMaskLambertConformal = True
   ;;res@gsnAddCyclic          = False
   ;;res@mpCenterLonF = 300.
   res@mpMinLonF =-90.;;-130.
   res@mpMaxLonF =  0.
   res@mpMinLatF = 45.;;20.
   res@mpMaxLatF = 85.

   resc = True
   resc@gsnMaximize = False
   resc@gsnDraw     = False      
   resc@gsnFrame    = False      
   resc@cnFillOn          = False
   resc@cnLinesOn         = True
   resc@cnLineLabelsOn    = False
   resc@cnInfoLabelOn     = False

;******************************************************
; begin loop
;******************************************************

 do n = 0,nfiles-1

   era5data = False
   merra2data = False
   if (n.eq.nera5) then
     era5data = True
   else if (n.eq.nmerra2) then
     merra2data = True
   end if 
   end if

   print("load:"+subdir_scrip+fname0(n))
   f = addfile(subdir_scrip + fname0(n),"r")
   ;;var2 = f->av
   ;;xp2 = f->xp
   ;;yp2 = f->yp
   blobid = f->blobid
   blobtc = f->blobtc
print("blobid: " + blobid(:,kframe))
print("blobtc: " + blobtc(:,kframe))
   delete(f)

   do p=0,3
     labels((n*4)+p) = tostring(dates(blobtc(p,kframe)))
   end do

   if (era5data) then
     files = systemfunc ("ls " + era5dir + "*.nc")
   else if (merra2data) then
     files = systemfunc ("ls " + merra2dir + "*.nc")
   else
     files = systemfunc ("ls " + dir + fname(n) + rdir + fname(n) + fdate)
   end if
   end if

   nf = dimsizes(files)
   f  = addfile (files(0),"r")
   dims = getfiledimsizes(f)
   dimnames = getfiledimnames(f)

   if (any(dimnames.eq."lon")) then
     print("history fields on lat-lon grid")
     latlon = True
   else
     print("history fields on unstructured grid")
     latlon = False
   end if

   lat = f->lat
   lon = f->lon
   delete(f)

   ntime = 0
   do m = 0,nf-1

    print("load: " + files(m))
    f  = addfile (files(m),"r")
    dims = getfiledimsizes(f)
    dimnames = getfiledimnames(f)
    tind= ind(dimnames.eq."time")
    dtime = dims(tind)
    delete([/dims,dimnames/])

    if (m.eq.0) then
      dims = dimsizes(blobtc)
      nens = dims(0)
      delete(dims)

      if (era5data) then
        vars = f->PRECT(0:nens-1,:,:)
      else if (merra2data) then
        vars = f->PRECTOT(0:nens-1,:,:)
      else
        if (latlon) then
          vars = f->PRECC(0:nens-1,:,:)
        else
          vars = f->PRECC(0:nens-1,:)
        end if   
      end if
      end if
      vars = 0.
    end if

    np = 0
    do p = 0,nens-1
      if ( (blobtc(p,kframe) .ge. ntime) .and. (blobtc(p,kframe) .lt. (ntime+dtime)) )
      print("blobtc(p,kframe): " + blobtc(p,kframe) + " ntime: " + ntime + " ntime+dtime: " + (ntime+dtime)) 
        ;;dates = f->date
        ;;time = f->time
        newtc= blobtc(p,kframe)-ntime
        ;;print("date: " + dates(newtc))
        ;;print("day: "  + time(newtc))

        ;;labels((n*nens)+p) = tostring(dates(newtc))

        if (era5data) then
          vars(p,:,:) = f->PRECT(newtc,:,:)
        else if (merra2data) then
          vars(p,:,:) = f->PRECTOT(newtc,:,:) ;;PRECTOTCORR
        else
          if (latlon) then
            vars(p,:,:) = f->PRECC(newtc,:,:)
            vartmp = f->PRECL(newtc,:,:)
            vars(p,:,:) = vars(p,:,:) + vartmp
            delete(vartmp)         
          else
            vars(p,:) = f->PRECC(newtc,:)
            vartmp = f->PRECL(newtc,:)
            vars(p,:) = vars(p,:) + vartmp
            delete(vartmp)
          end if
        end if
        end if
        np = np + 1
        ;;delete([/dates,time/])
      end if 
    end do

    ntime = ntime + dtime
    delete(f)

    if (np.eq.nens) then
      continue
    end if

   end do 
   delete(files)

   if (era5data) then
     vars = vars*era5factor
   else if (merra2data) then
     vars = vars*merra2factor
   else
     vars = vars*factor
   end if
   end if

   if (latlon) then
     vars!1 = "lat"
     vars&lat = lat
     vars&lat@units = "degrees_north"
     vars!2 = "lon"
     vars&lon = lon
     vars&lon@units = "degrees_east"
   else
     res@sfXArray = lon
     res@sfYArray = lat
   end if
   delete([/lat,lon/])

   print("max: " + max(vars))
   print("min: " + min(vars))

   ;;set contour levels
   ;maxcn  =150.
   ;mincn  =  0.
   ;cnlevs = 25.
   ;res@cnLevelSelectionMode = "ManualLevels"
   ;res@cnMinLevelValF  = mincn
   ;res@cnMaxLevelValF  = maxcn
   ;res@cnLevelSpacingF = (maxcn-mincn)/cnlevs

   res@cnLevelSelectionMode = "ExplicitLevels"
   res@cnLevels = (/0.5,1.,2.,5.,10.,15.,20.,30.,40.,50.,75.,100.,125.,150.,175.,200.,225.,250./)

   if (cellfill) then
     res@cnFillMode = "CellFill"
     vf = addfile(griddir(n)+scrip(n),"r")
     res@sfXCellBounds = vf->grid_corner_lon
     res@sfYCellBounds = vf->grid_corner_lat
     delete(vf)
   else
     res@cnFillMode = "AreaFill"
   end if 

   delete(vars@standard_name)
   delete(vars@long_name)
   delete(vars@units)

   do p = 0,nens-1
    
     pid = n*nens + p

     if (latlon) then
       plot(pid) = gsn_csm_contour_map(wks,vars(p,:,:),res)
     else
       plot(pid) = gsn_csm_contour_map(wks,vars(p,:),res)
     end if
   end do
   delete(vars)

   if (.not.latlon) then
     delete([/res@sfXArray,res@sfYArray/])
   end if

   if (cellfill) then
     delete([/res@sfXCellBounds,res@sfYCellBounds/])
   end if

 end do;;n

;;drawNDCGrid(wks)
;*****************************************************
; Plot panel
;*****************************************************

  resP                  = True                   ; modify the panel plot
  resP@gsnFrame         = False                  ; don't advance panel plot
  ;;resP@gsnPaperOrientation = "landscape"

  resP@gsnPanelFigureStrings = labels
  resP@gsnPanelFigureStringsFontHeightF = 0.005
  resP@gsnPanelFigureStringsPerimOn     = True
  resP@gsnPanelFigureStringsBackgroundFillColor = "white"
  resP@amJust           = "TopLeft"

  resP@gsnPanelLabelBar = False
  resP@gsnPanelLabelBar = True
  resP@lbBoxSeparatorLinesOn= False
  resP@lbTitleString              = lbtitle
  resP@lbTitlePosition            = "Bottom"
  resP@lbTitleFontHeightF         = 0.015
  resP@lbLabelFontHeightF         = 0.01
  resP@pmLabelBarOrthogonalPosF   = -0.02

  resP@gsnPanelBottom     = 0.05

  gsn_panel(wks,plot,(/nfiles,4/),resP)               ; now draw as one plot

frame(wks)

end

