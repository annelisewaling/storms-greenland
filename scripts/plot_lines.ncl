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
  p0 = 1000.d2
  secpday = 60.*60.*24;
  pi = 3.14159265358979
  g = 9.80616
  Re = 6371.22e3
  cp = 1004.
  Rd = 287.
;****************************************
  subdir = "/glade/campaign/cgd/amp/aherring/annelise_backup/TempestExtremes/data/lines2/"
  subdir_scrip = "../data/lines/"


  fname1 = (/"test_geolines_ESMF_COUNT_aint_f19.nc", \
             "test_geolines_ESMF_COUNT_aint_f09.nc", \
             "test_geolines_ESMF_COUNT_aint_ne30pg2.nc", \
             "test_geolines_ESMF_COUNT_aint_ne30pg3.nc", \
             "test_geolines_ESMF_COUNT_aint_ARCTIC.nc", \
             "test_geolines_ESMF_COUNT_aint_ARCTICGRIS.nc", \
             "geolines_PRECTOTCORR_COUNT_aint_ERA5.nc" , \
             "geolines_PRECTOTCORR_COUNT_aint_MERRA2.nc"/)
  fname1b = (/"test_geolines_ESMF_TCOUNT_aint_f19.nc", \
             "test_geolines_ESMF_TCOUNT_aint_f09.nc", \
             "test_geolines_ESMF_TCOUNT_aint_ne30pg2.nc", \
             "test_geolines_ESMF_TCOUNT_aint_ne30pg3.nc", \
             "test_geolines_ESMF_TCOUNT_aint_ARCTIC.nc", \
             "test_geolines_ESMF_TCOUNT_aint_ARCTICGRIS.nc", \
             "geolines_PRECTOTCORR_TCOUNT_aint_ERA5.nc", \   
             "geolines_PRECTOTCORR_TCOUNT_aint_MERRA2.nc"/)
  fname2 = (/"test_geolines_ESMF_TAREA_aint_f19.nc",\
             "test_geolines_ESMF_TAREA_aint_f09.nc",\
             "test_geolines_ESMF_TAREA_aint_ne30pg2.nc",\
             "test_geolines_ESMF_TAREA_aint_ne30pg3.nc",\
             "test_geolines_ESMF_TAREA_aint_ARCTIC.nc",\
             "test_geolines_ESMF_TAREA_aint_ARCTICGRIS.nc",\
             "geolines_PRECTOTCORR_TAREA_aint_ERA5.nc", \
             "geolines_PRECTOTCORR_TAREA_aint_MERRA2.nc"/)
  fname3 = (/"test_geolines_ESMF_AREA_aint_f19.nc", \
             "test_geolines_ESMF_AREA_aint_f09.nc",\
             "test_geolines_ESMF_AREA_aint_ne30pg2.nc",\
             "test_geolines_ESMF_AREA_aint_ne30pg3.nc",\
             "test_geolines_ESMF_AREA_aint_ARCTIC.nc",\
             "test_geolines_ESMF_AREA_aint_ARCTICGRIS.nc",\
             "geolines_PRECTOTCORR_AREA_aint_ERA5.nc", \
             "geolines_PRECTOTCORR_AREA_aint_MERRA2.nc"/)

  fname4 = (/"geolines_PRECTOTCORR_TIVT_aint_f19.nc", \
             "geolines_PRECTOTCORR_TIVT_aint_f09.nc", \
             "geolines_PRECTOTCORR_TIVT_aint_ne30pg2.nc", \
             "geolines_PRECTOTCORR_TIVT_aint_ne30pg3.nc", \
             "geolines_PRECTOTCORR_TIVT_aint_ARCTICGRIS.nc", \
             "geolines_PRECTOTCORR_TIVT_aint_ARCTIC.nc", \
             "geolines_PRECTOTCORR_TIVT_aint_ERA5.nc", \
             "geolines_PRECTOTCORR_TIVT_aint_MERRA2.nc"/)

  fname5 = (/"geolines_PRECTOTCORR_TIVT_amean_f19.nc", \
             "geolines_PRECTOTCORR_TIVT_amean_f09.nc", \
             "geolines_PRECTOTCORR_TIVT_amean_ne30pg2.nc", \
             "geolines_PRECTOTCORR_TIVT_amean_ne30pg3.nc", \
             "geolines_PRECTOTCORR_TIVT_amean_ARCTICGRIS.nc", \
             "geolines_PRECTOTCORR_TIVT_amean_ARCTIC.nc", \
             "geolines_PRECTOTCORR_TIVT_amean_ERA5.nc", \
             "geolines_PRECTOTCORR_TIVT_amean_MERRA2.nc"/)

  nfiles = dimsizes(fname4)

  labels = (/"f19", \
             "f09", \
             "ne30pg2", \
             "ne30pg3", \
             "ARCTIC", \
             "ARCTICGRIS", \
             "ERA5" , \
             "MERRA2"/)
  


  colors = (/"blue","deepskyblue","green4","green","darkorchid4","magenta","black", "grey40"/)

;******************************************************

  clmframes = (/-5.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,5./)
  ;camframes = (/-2.,-1.75,-1.5,-1.25,-1.,-0.75,-0.50,-0.25,0.,0.25,0.50,0.75,1.,1.25,1.5,1.75,2./)
  ;;camframes = (/-1.25,-1.,-0.75,-0.50,-0.25,0.,0.25,0.50,0.75,1.,1.25/)
  camframes = fspan(-5.,5.,41)
  ;camframes = (/-1.,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1./)
  dtclm = clmframes(1)-clmframes(0)
  dtcam = camframes(1)-camframes(0)

  xtime1 = camframes
  xtime2 = camframes
  xtime3 = camframes
  xtime4 = camframes
  xtime5 = camframes

  yaxis1b = "# ARs overlapping GrIS"
  ymin1 =   0.
  ymax1 = 700.
  factor1 = 1.
  
  yaxis1 = "total # ARs"

  yaxis2 = "AR Area (10~S~10~N~ m~S~2~N~)"
  ymin2 = 120.
  ymax2 = 220.
  factor2 = 1.e-10

  ;yaxis3 = "% AR overlap area/total AR size"
  yaxis3 = "AR Overlap Area (10~S~10~N~ m~S~2~N~)"
  ymin3 = 0.0
  ymax3 = 4.5
  factor3 = 1.e-10

  xaxis = "Days relative to max overlap with GrIS"
  
  yaxis4 = "IVT (Gt s-1)"
  ymin4 = 0.
  ymax4 = 1500.
  factor4 = 1.e-12

  yaxis5 = "IVT (kg m-1 s-1)"
  ymin5 = 500.
  ymax5 = 700.
  factor5 = 1

  xmin1 = -2.
  xmax1 =  2.
  xmin2 = -2.5
  xmax2 =  2.5
  xmin3 = -2.5
  xmax3 =  2.5
  xmin4 = -2.
  xmax4 = 2.
  xmin5 = -2.
  xmax5 = 2.

  pltNam = "temp_lines"

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

   ;res@gsnPaperMargin = 5
   res@gsnPanelBottom = 1

   res@vpWidthF            = 0.7           ; change aspect ratio of plot
   res@vpHeightF           = 0.26

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
   res@tmYLLabelFontHeightF = 0.015
   
   res@gsnYRefLine = 0.
   
   ;;; transparent polygons
   gsres                = True
   gsres@gsFillOpacityF = 0.1;;0.3
   gsres@gsLineOpacityF = 0.1
   gsres@tfPolyDrawOrder = "PreDraw"
 

;******************************************************
; begin loop
;******************************************************

 do n = 0,nfiles-1

   print("load:"+subdir_scrip+fname1(n))
   f = addfile(subdir_scrip + fname1(n),"r")
   var1 = f->av
   xp1 = f->xp
   yp1 = f->yp
   delete(f)   

   var1 = factor1*var1
   yp1 = factor1*yp1

   print("load:"+subdir_scrip+fname1b(n))
   f = addfile(subdir_scrip + fname1b(n),"r")
   var1b = f->av
   xp1b = f->xp
   yp1b = f->yp
   delete(f)

   var1b = factor1*var1b

   print("load:"+subdir_scrip+fname2(n))
   f = addfile(subdir_scrip + fname2(n),"r")
   var2 = f->av
   xp2 = f->xp
   yp2 = f->yp
   delete(f)

   var2 = factor2*var2
   yp2 = factor2*yp2
  
   ;xtime2@_FillValue = default_fillvalue("float") 
   ;xtime2 = where(abs(xtime2).gt.2.25,xtime2@_FillValue,xtime2)

   print("load:"+subdir_scrip+fname3(n))
   f = addfile(subdir_scrip + fname3(n),"r")
   var3 = f->av
   xp3 = f->xp
   yp3 = f->yp
   delete(f)

   var3 = factor3*var3
   yp3 = factor3*yp3

   ;create new figure which looks at percentage of AR overlap area/AR total area
   ;var4 = (var3/var2)*100
   ;yp4 = (yp3/yp2)*100
   
   ;var4(:15) = var4@_FillValue
   ;var4(25:) = var4@_FillValue
   ;yp4(:15) = var4@_FillValue
   ;yp4(25:) = var4@_FillValue

   var3(:15) = var3@_FillValue
   var3(25:) = var3@_FillValue


   xtime3@_FillValue = default_fillvalue("float")
   xtime3 = where(abs(xtime3).gt.2.5,xtime3@_FillValue,xtime3)

   print("load:"+subdir_scrip+fname4(n))
   f = addfile(subdir_scrip + fname4(n),"r")
   var4 = f->av
   xp4 = f->xp
   yp4 = f->yp
   delete(f)

   var4 = factor4*var4
   yp4 = factor4*yp4

   print("load:"+subdir_scrip+fname5(n))
   f = addfile(subdir_scrip + fname5(n),"r")
   var5 = f->av
   xp5 = f->xp
   yp5 = f->yp
   delete(f)

   res@xyDashPatterns = 0
   res@xyLineThicknessF  = 4
   res@xyLineColors = colors(n)

   if (n.eq.0) then

     res@gsnLeftStringFontHeightF    = 0.002 

     res@tiYAxisString   = yaxis1
     res@trYMinF         = ymin1
     res@trYMaxF         = ymax1
     res@trXMinF         = xmin1
     res@trXMaxF         = xmax1
     res@xyDashPatterns = 0
     plot(0)  = gsn_csm_xy(wks,xtime1,var1b,res)

     ;res@tiXAxisString   = xaxis     

     res@tiYAxisString   = yaxis1b
     res@trYMinF         = ymin1
     res@trYMaxF         = ymax1
     res@trXMinF         = xmin1
     res@trXMaxF         = xmax1
     res@xyDashPatterns = 0
     plot(1)  = gsn_csm_xy(wks,xtime1,var1,res) 
  
     ;delete(res@tiXAxisString);  = " "     

     ;res@xyDashPatterns = 1
     ;tmp1b = gsn_csm_xy(wks,xtime1,var1b,res)
     ;overlay(plot(0),tmp1b)
     ;delete(tmp1b)
     res@xyDashPatterns = 0

     res@tiYAxisString   = yaxis2
     res@trYMinF         = ymin2
     res@trYMaxF         = ymax2
     res@trXMinF         = xmin2
     res@trXMaxF         = xmax2
     plot(2)  = gsn_csm_xy(wks,xtime2,var2,res)

     res@tiXAxisString   = xaxis

     res@tiYAxisString   = yaxis3
     res@trYMinF         = ymin3
     res@trYMaxF         = ymax3
     res@trXMinF         = xmin3
     res@trXMaxF         = xmax3
     plot(3)  = gsn_csm_xy(wks,xtime3,var3,res)
    
     res@tiYAxisString   = yaxis4
     res@trYMinF         = ymin4
     res@trYMaxF         = ymax4
     res@trXMinF         = xmin4
     res@trXMaxF         = xmax4
     ;plot(0) = gsn_csm_xy(wks,xtime4,var4,res)

     res@tiYAxisString   = yaxis5
     res@trYMinF         = ymin5
     res@trYMaxF         = ymax5
     res@trXMinF         = xmin5
     res@trXMaxF         = xmax5
     ;plot(0) = gsn_csm_xy(wks,xtime5,var5,res)

     delete(res@tiXAxisString)

   else
     res@xyDashPatterns = 0
     tmp1 = gsn_csm_xy(wks,xtime1,var1b,res)
     overlay(plot(0),tmp1)
     delete(tmp1)    

     res@xyDashPatterns = 0
     tmp1b = gsn_csm_xy(wks,xtime1,var1,res)
     overlay(plot(1),tmp1b)
     delete(tmp1b)
     res@xyDashPatterns = 0

     tmp2 = gsn_csm_xy(wks,xtime2,var2,res)
     overlay(plot(2),tmp2)
     delete(tmp2)

     tmp3 = gsn_csm_xy(wks,xtime3,var3,res)
     overlay(plot(3),tmp3)
     delete(tmp3)

     tmp4 = gsn_csm_xy(wks,xtime4,var4,res)
     ;overlay(plot(0),tmp4)
     delete(tmp4)

     tmp5 = gsn_csm_xy(wks,xtime5,var5,res)
     ;overlay(plot(0),tmp5)
     delete(tmp5)

   end if
   delete([/var1,var2,var3,var1b,var4,var5/])

   ;;now plot polygon shading

   xp1 = xp1 - 1
   m1 = dimsizes(xp1)
   xxp1 = new( (/m1/), "float")
   do m = 0,m1-1
     xxp1(m) = camframes(tointeger(xp1(m)))
   end do

   xp1b = xp1b - 1
   m1b = dimsizes(xp1b)
   xxp1b = new( (/m1b/), "float")
   do m = 0,m1b-1
     xxp1b(m) = camframes(tointeger(xp1b(m)))
   end do

   xp2 = xp2 - 1
   m2 = dimsizes(xp2)
   xxp2 = new( (/m2/), "float")
   do m = 0,m2-1
     xxp2(m) = camframes(tointeger(xp2(m)))
   end do

   ;xxp2@_FillValue = default_fillvalue("float")
   ;xxp2 = where(abs(xxp2).gt.2.25,xxp2@_FillValue,xxp2)

   xp3 = xp3 - 1
   m3 = dimsizes(xp3)
   xxp3 = new( (/m3/), "float")
   do m = 0,m3-1
     xxp3(m) = camframes(tointeger(xp3(m)))
   end do

   xp4 = xp4 - 1
   m4 = dimsizes(xp4)
   xxp4 = new( (/m4/), "float")
   do m = 0,m4-1
     xxp4(m) = camframes(tointeger(xp4(m)))
   end do

   xp5 = xp5 - 1
   m5 = dimsizes(xp5)
   xxp5 = new( (/m5/), "float")
   do m = 0,m5-1
     xxp5(m) = camframes(tointeger(xp5(m)))
   end do
  
   xxp3@_FillValue = default_fillvalue("float")
   xxp3 = where(abs(xxp3).gt.2.5,xxp3@_FillValue,xxp3)

   yyp1 = yp1

   gsres@gsFillColor    = colors(n)
   
   polystr = unique_string("hum")
   plot@$polystr$ = gsn_add_polygon(wks,plot(1),xxp1,yyp1,gsres)
   delete([/yyp1,xxp1,xp1,yp1/])

   yyp1b = yp1b
   
   polystr = unique_string("hum")
   plot@$polystr$ = gsn_add_polygon(wks,plot(0),xxp1b,yyp1b,gsres)
   delete([/yyp1b,xxp1b,xp1b,yp1b/]) 
 
   yyp2 = yp2
   
   polystr = unique_string("hum")
   plot@$polystr$ = gsn_add_polygon(wks,plot(2),xxp2,yyp2,gsres)
   delete([/xxp2,yyp2,xp2,yp2/])

   yyp3 = yp3

   xxp3(:16) = xxp3@_FillValue
   xxp3(26:57) = xxp3@_FillValue
   xxp3(67:) = xxp3@_FillValue    

   ;print(xxp3)

   polystr = unique_string("hum")
   plot@$polystr$ = gsn_add_polygon(wks,plot(3),xxp3,yyp3,gsres)
   
   yyp4 = yp4

   polystr = unique_string("hum")
   ;plot@$polystr$ = gsn_add_polygon(wks,plot(0),xxp4,yyp4,gsres)

   yyp5 = yp5
   
   polystr = unique_string("hum")
   ;plot@$polystr$ = gsn_add_polygon(wks,plot(0),xxp5,yyp5,gsres)

   delete([/xxp3,yyp3,xp3,yp3,yp4/]) 


 end do;;n


;*****************************************************
; Add legend
;*****************************************************
 txres = True
 txres@txFontHeightF = 0.012
 txres@txJust = "CenterLeft"

 resln             = True
 resln@gsLineThicknessF = res@xyLineThicknessF
 resln@lgBoxBackground = 1

 ;yy = (/0.74,0.728,0.716,0.704,0.692,0.68,0.668,0.656/)-0.51;-0.47;;+0.23
 xx = 0.28
 dx = 0.03
 ;do n = 0,nfiles-1
 ;  resln@gsLineColor = colors(n)
 ;  gsn_polyline_ndc(wks,(/xx,xx+dx/),(/yy(n),yy(n)/),resln)
 ;  gsn_text_ndc(wks,"  " + labels(n),xx+dx,yy(n),txres)
 ;end do
 yy = (/0.74,0.725,0.71,0.695,0.68,0.665,0.65,0.635/)-0.52
 ;yy = (/0.74,0.72,0.70,0.68,0.66,0.64,0.62, 0.60/)-0.10
 do n = 0,nfiles-1
   resln@gsLineColor = colors(n)
   gsn_polyline_ndc(wks,(/xx,xx+dx/),(/yy(n),yy(n)/),resln)
   gsn_text_ndc(wks,"  " + labels(n),xx+dx,yy(n),txres)
 end do

;;drawNDCGrid(wks)
;*****************************************************
; Plot panel
;*****************************************************

  resP                  = True                   ; modify the panel plot
  resP@gsnFrame         = False                  ; don't advance panel plot

  resP@gsnPanelBottom = 0.02
  resP@gsnPanelYWhiteSpacePercent = 5

  ;;resP@txString = plotTitle
  resP@gsnPanelFigureStrings = (/"a.","b.","c.","d."/)
  resP@gsnPanelFigureStringsFontHeightF = 0.015
  ;resP@gsnPanelFigureStringsPerimOn     = True
  ;resP@gsnPanelFigureStringsBackgroundFillColor = "white"
  resP@amJust           = "BottomLeft"
  resP@gsnPanelLeft = 0.05
  resP@gsnPanelRight = 0.95
  ;resP@gsnPanelBottom   = 0.1

  gsn_panel(wks,plot,(/4,1/),resP)

frame(wks)

end
