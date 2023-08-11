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

  rdir = "/atm/hist/"
  dir = "/glade/campaign/cgd/amp/aherring/archive/"

  fname1 = (/"cam6_2_022.se_FHIST_f19_f19_mg17_900pes_200506_mg3-Nx5yrs", \
            "cam6_2_022.se_FHIST_f09_f09_mg17_1800pes_200507_mg3-Nx5yrs", \
            "cam6_2_022.se_FHIST_ne30pg2_ne30pg2_mg17_1800pes_200507_mg3-Nx5yrs", \
            "cam6_2_022.se_FHIST_ne30pg3_ne30pg3_mg17_1800pes_200507_mg3-Nx5yrs", \
            "cam6_2_022.se_FHIST_ne0np4.ARCTIC.ne30x4_mt12_7680pes_200507_mg3-Nx2yrs", \
            "cam6_2_022.se_FHIST_ne0np4.ARCTICGRIS.ne30x8_mt12_7680pes_200510_mg3-Nx1yrs",\
	    "MERRA2_orig_res_19990325", \
	    "adaptor.mars.internal-1676046438.1816635-19214-16-99297910-d736-4d29-b0a0-90e48b389e11"/)

  runnams1 = (/"a. f19", \
              "b. f09", \
              "c. ne30pg2", \
              "d. ne30pg3", \
              "e. ARCTIC", \
              "f. ARCTICGRIS", \
	      "g. MERRA2", \
	      "h. ERA5"/)

  nfiles = dimsizes(fname1)

  fincl = "*"

  VAR = "PHIS"
  factor = 1./g

  ;;set contour levels
  maxcn1  = 3200.
  mincn1  = 0.
  maxcn2  = 1000.
  mincn2  =-1000.
  cnlevs1 = 40.
  cnlevs2 = 50.

  pltNam = "temp_topo"

  cellfill = True

  griddir1 = "../data/scrip/"

  scrip1 = (/"fv1.9x2.5_141008.nc", \
            "fv0.9x1.25_scrip_141008.nc", \
            "ne30pg2_scrip_170608.nc", \
            "ne30pg3_scrip_170611.nc", \
            "ne0ARCTICne30x4_scrip_191212.nc", \
            "ne0ARCTICGRISne30x8_scrip_191209.nc", \
	    "MERRA2_scrip.nc", \
	    "ERA5_025_221025.nc"/)

;******************************************************
; create plot
;******************************************************
  wkstype = "pdf"
  wks = gsn_open_wks(wkstype,pltNam)               ; open workstation

  gsn_define_colormap(wks,"BkBlAqGrYeOrReViWh200")
  fname = fname1
  griddir = griddir1
  scrip = scrip1
  runnams = runnams1
  maxcn = maxcn1
  mincn = mincn1
  cnlevs = cnlevs1

  plot = new( (/nfiles/), "graphic")

   res             = True
   res@gsnMaximize = False
   res@gsnDraw     = False      ; Will draw later, after overlaying
   res@gsnFrame    = False      ; all plots

   res@cnFillOn          = False               ; turn on color fill
   res@cnLinesOn         = True
   res@cnLineLabelsOn    = True

   res@cnFillOn          = True               ; turn on color fill
   res@cnLinesOn         = False
   res@cnLineLabelsOn    = False

   res@gsnSpreadColors   = True               ; use full range of color map
   res@cnInfoLabelOn        = False
   res@lbLabelBarOn         = False
   res@tiMainFontHeightF = 0.0007

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

   res@mpProjection = "LambertConformal";;"Orthographic";;"Mollweide"
   res@mpLambertParallel1F = 85
   res@mpLambertParallel2F = 58
   res@mpLambertMeridianF  = -45
   res@mpLimitMode       = "LatLon"
   res@mpMinLonF = -56.4
   res@mpMaxLonF = -30.
   res@mpMinLatF = 59.4;;58.
   res@mpMaxLatF = 85;;85.

;******************************************************
; begin loop
;******************************************************

 do n = 0,nfiles-1

   newdir = "../data/topo/"
   files = systemfunc ("ls " + newdir + fname(n) + fincl)

   print("load file: " + files(0))
   f  = addfile (files(0),"r")
   if (n.eq.nfiles-1) then
     var = f->GEOPOT
   else
     var = f->$VAR$
   end if
   var = var*factor
   delete(files)

   dims = dimsizes(dimsizes(var))
   if (dims.eq.3) then
     latlon = True

     if (n.eq.nfiles-1) then
       lat = var&latitude
       lon = var&longitude
     else
       lat = var&lat
       lon = var&lon
     end if

     tmp = var(0,:,:)
     delete(var)
     var = tmp
     delete(tmp)

     var!0 = "lat"
     var&lat = lat
     var&lat@units = "degrees_north"
     var!1 = "lon"
     var&lon = lon
     var&lon@units = "degrees_east"
     delete([/lat,lon/])
   else
     latlon = False

     tmp = var(0,:)
     delete(var)
     var = tmp
     delete(tmp)

     lat1d = f->lat
     lon1d = f->lon
     res@sfXArray = lon1d
     res@sfYArray = lat1d
     delete([/lat1d,lon1d/])
   end if
   delete(f)

   print("max: " + max(var))
   print("min: " + min(var))

   res@cnLevelSelectionMode = "ManualLevels"
   res@cnMinLevelValF  = mincn
   res@cnMaxLevelValF  = maxcn
   res@cnLevelSpacingF = (maxcn-mincn)/cnlevs

   if (cellfill) then
     res@cnFillMode = "CellFill"
     vf = addfile(griddir+scrip(n),"r")
     res@sfXCellBounds = vf->grid_corner_lon
     res@sfYCellBounds = vf->grid_corner_lat
     delete(vf)
   else
     res@cnFillMode = "AreaFill"
   end if 

   delete(var@long_name)
   delete(var@units)
   delete(var@standard_name)

   res@gsnCenterStringFontHeightF = 0.01
   plot(n) = gsn_csm_contour_map(wks,var,res)
   delete(var)

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

  resP@gsnPanelFigureStrings = runnams
  ;resP@gsnPanelFigureStrings = (/"a.","b.","c.","d.","e.","f.","g."/)
  resP@gsnPanelFigureStringsFontHeightF = 0.009
  resP@gsnPanelFigureStringsPerimOn     = True
  resP@gsnPanelFigureStringsBackgroundFillColor = "white"
  resP@amJust           = "TopLeft"

  resP@gsnPanelLabelBar = False
  resP@gsnPanelLabelBar = True
  resP@lbBoxSeparatorLinesOn= False
  ;;resP@lbTitleString              = lbtitle
  ;;resP@lbTitlePosition            = "Bottom"
  ;;resP@lbLabelPosition            = "Vertical"
  ;;resP@lbLabelAngleF              = 90.
  ;;resP@lbTitleDirection           = "Down"
  resP@lbOrientation              = "Vertical"
  resP@lbTitleFontHeightF         = 0.015
  resP@lbLabelFontHeightF         = 0.01
  resP@pmLabelBarOrthogonalPosF   = 0.00

  ;;resP@pmLabelBarHeightF = 0.02
  resP@pmLabelBarWidthF = 0.05

  resP@gsnPanelTop = .95
  resP@gsnPanelBottom = .60
  resP@gsnPanelLeft = 0.05
  resP@gsnPanelRight = 0.95

  gsn_panel(wks,plot,(/1,nfiles/),resP)               ; now draw as one plot
  
frame(wks)

end

