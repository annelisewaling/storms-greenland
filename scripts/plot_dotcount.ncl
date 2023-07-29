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

  title = "1979-1998, JJA, IVT threshold = -30k"

  dir = "/glade/campaign/cgd/amp/aherring/annelise_backup/adams_folder/data/mapcount/"
  dir = "../data/mapcount/"

  season = "JJA."
  ;;season = "JJA.-50k."

  fname1 = (/"mapcount_f19."+season+"ESMF_f19.nc", \
	    "mapcount_f09."+season+"ESMF_f19.nc", \
            "mapcount_ne30pg2."+season+"ESMF_f19.nc", \
            "mapcount_ne30pg3."+season+"ESMF_f19.nc", \
            "mapcount_ARCTIC."+season+"ESMF_f19.nc", \
            "mapcount_ARCTICGRIS."+season+"ESMF_f19.nc", \
            "mapcount_MERRA2."+season+"ESMF_f19.nc", \
            "mapcount_ERA5."+season+"ESMF_f19.nc"/)
  fname2 = (/"mapcount_f19."+season+"ESMF_pg2.nc", \
            "mapcount_f09."+season+"ESMF_pg2.nc", \
            "mapcount_ne30pg2."+season+"ESMF_pg2.nc", \
            "mapcount_ne30pg3."+season+"ESMF_pg2.nc", \
            "mapcount_ARCTIC."+season+"ESMF_pg2.nc", \
            "mapcount_ARCTICGRIS."+season+"ESMF_pg2.nc", \
            "mapcount_MERRA2."+season+"ESMF_pg2.nc", \
            "mapcount_ERA5."+season+"ESMF_pg2.nc"/)
  fname3 = (/"mapcount_f19."+season+"TR_f19.nc", \
            "mapcount_f09."+season+"TR_f19.nc", \
            "mapcount_ne30pg2."+season+"TR_f19.nc", \
            "mapcount_ne30pg3."+season+"TR_f19.nc", \
            "mapcount_ARCTIC."+season+"TR_f19.nc", \
            "mapcount_ARCTICGRIS."+season+"TR_f19.nc", \
            "mapcount_MERRA2."+season+"TR_f19.nc", \
            "mapcount_ERA5."+season+"TR_f19.nc"/)
  fname4 = (/"mapcount_f19."+season+"TR_pg2.nc", \
            "mapcount_f09."+season+"TR_pg2.nc", \
            "mapcount_ne30pg2."+season+"TR_pg2.nc", \
            "mapcount_ne30pg3."+season+"TR_pg2.nc", \
            "mapcount_ARCTIC."+season+"TR_pg2.nc", \
            "mapcount_ARCTICGRIS."+season+"TR_pg2.nc", \
            "mapcount_MERRA2."+season+"TR_pg2.nc", \
            "mapcount_ERA5."+season+"TR_pg2.nc"/)
  
  nfiles = dimsizes(fname1)

  runnams = (/"f19", \
              "f09", \
              "ne30pg2", \
              "ne30pg3", \
              "ARCTIC", \
              "ARCTICGRIS", \
	      "MERRA2", \
	      "ERA5"/)

  VAR = "mapcount"
  factor = 1.

  colors = (/"blue","red4","deepskyblue","red"/)

  pltNam = "temp_contours"

;******************************************************
; create plot
;******************************************************
  wkstype = "pdf"
  wks = gsn_open_wks(wkstype,pltNam)               ; open workstation

  plot = new( (/nfiles/), "graphic")

   res             = True
   res@gsnMaximize = False
   res@gsnDraw     = False      ; Will draw later, after overlaying
   res@gsnFrame    = False      ; all plots

;   res@vpWidthF  = 0.3
;   res@vpHeightF = 0.8

   res@mpGridAndLimbOn   = False           ; turn on lat/lon lines
   res@mpGridLatSpacingF = 10.
   res@mpGridLonSpacingF = 30.
   ;res@mpGridLineColor      = -1
   res@mpPerimOn         = True               ; turn off perimeter
   res@mpFillOn         = True
   ;;res@mpOutlineBoundarySets     = "NoBoundaries"

;   res@mpProjection = "Orthographic"
;   res@mpCenterLatF      =   45.
;   res@mpCenterLonF      =  300.

   res@mpProjection = "LambertConformal"

;   res@mpLambertParallel1F = 85
;   res@mpLambertParallel2F = 20
;   res@mpLambertMeridianF  = -45
;   res@mpLimitMode       = "LatLon"


   res@gsnMaskLambertConformal = True
   res@gsnAddCyclic          = False
   ;;res@mpCenterLonF = 300.
   res@mpMinLonF =-130.
   res@mpMaxLonF =  0.
   res@mpMinLatF = 20.
   res@mpMaxLatF = 85.

   ;res@mpOutlineBoundarySets   = "AllBoundaries"
   ;res@mpDataSetName           = "Earth..3"


   resg = True

;******************************************************
; begin loop
;******************************************************

 do n = 0,nfiles-1

   plot(n) = gsn_csm_map(wks,res)

   fnames = (/fname1(n),fname2(n),fname3(n),fname4(n)/)
   nmems=4

   do m = 0,nmems-1   

     if (n.eq.0 .and. m.eq.2) then
       continue
     else if (n.eq.2 .and. m.eq.3) then
       continue
     end if
     end if

     print("load file: " + fnames(m))
     f  = addfile (dir+fnames(m),"r")
     var = f->$VAR$
     lat = f->lat
     lon = f->lon

     dimnames = getfiledimnames(f)
     delete(f)

     if (any(dimnames.eq."lon")) then
       latlon = True
     else
       latlon = False
     end if
     delete(dimnames)

     resg@gsMarkerColor = colors(m)
     resg@gsMarkerIndex = 1
     ;;resg@gsMarkerSizeF = 0.01
     ;;resg@gsMarkerThicknessF = 6

     dims = dimsizes(var)
     if (latlon) then
       ni = dims(0)
       nj = dims(1)
       do i=0,ni-1
         do j=0,nj-1
           if (var(i,j).eq.0) then
             continue
           end if
           ;;add marker         
           resg@gsMarkerSizeF = 0.005*var(i,j)
           pNam = unique_string("tmp")
           plot@$pNam$ = gsn_add_polymarker(wks,plot(n),lon(j),lat(i),resg)
         end do
       end do
     else
       ni = dims(0)
       do i = 0,ni-1
         if (var(i).eq.0) then
           continue
         end if
         ;;add marker
         resg@gsMarkerSizeF = 0.005*var(i)
         pNam = unique_string("tmp")
         plot@$pNam$ = gsn_add_polymarker(wks,plot(n),lon(i),lat(i),resg)
       end do
     end if
     delete([/dims,var,lat,lon/])
    
   end do;;m
   delete(fnames)

 end do;;n

;;drawNDCGrid(wks)
;*****************************************************
; Plot panel
;*****************************************************

  resP                  = True                   ; modify the panel plot
  resP@gsnFrame         = False                  ; don't advance panel plot

  resP@gsnPanelFigureStrings = runnams
  resP@gsnPanelFigureStringsFontHeightF = 0.015
  ;resP@gsnPanelFigureStringsPerimOn     = True
  resP@gsnPanelFigureStringsBackgroundFillColor = "white"
  resP@amJust           = "TopLeft"

  ;resP@gsnPanelLeft = 0.05
  ;resP@gsnPanelRight = 0.95

  resP@txString = title
  gsn_panel(wks,plot,(/4,2/),resP)               ; now draw as one plot

frame(wks)

end

