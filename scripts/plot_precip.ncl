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

  inttop = True
  intbot = True

  melt = False

  subdir = "/glade/campaign/cgd/amp/aherring/annelise_backup/TempestExtremes/data/lines2/"
  subdir_scrip = "../data/precip/"
     

fname11 = (/"geolines_ESMF_PRECT_amean_f19.nc", \ 
            "geolines_ESMF_PRECT_amean_f09.nc", \
            "geolines_ESMF_PRECT_amean_ne30pg2.nc", \
            "geolines_ESMF_PRECT_amean_ne30pg3.nc", \
            "geolines_ESMF_PRECT_amean_ARCTIC.nc", \
            "geolines_ESMF_PRECT_amean_ARCTICGRIS.nc", \
            "geolines_PRECTOTCORR_PRECT_amean_ERA5.nc", \
            "geolines_PRECTOTCORR_PRECT_amean_MERRA2.nc"/) 
            ;"geolines_TEST_PRECT_amean_MERRA2.nc"/)

fname12 = (/"geolines_ESMF_PRECT_aint_f19.nc", \
            "geolines_ESMF_PRECT_aint_f09.nc", \
            "geolines_ESMF_PRECT_aint_ne30pg2.nc", \
            "geolines_ESMF_PRECT_aint_ne30pg3.nc", \
            "geolines_ESMF_PRECT_aint_ARCTIC.nc", \
            "geolines_ESMF_PRECT_aint_ARCTICGRIS.nc", \
            "geolines_PRECTOTCORR_PRECT_aint_ERA5.nc", \ 
            "geolines_PRECTOTCORR_PRECT_aint_MERRA2.nc"/)

nfiles = dimsizes(fname11)


  if (melt) then
    labels = (/"f19", \
               "f09", \
               "ne30pg2", \
               "ne30pg3", \
               "ARCTIC", \
               "ARCTICGRIS", \
               "ERA5"/)
    labels2 = (/"f19", \
               "f09", \
               "ne30pg2", \
               "ne30pg3", \
               "ARCTIC", \
               "ARCTICGRIS"/)
   else 
     labels = (/"f19", \
               "f09", \
               "ne30pg2", \
               "ne30pg3", \
               "ARCTIC", \
               "ARCTICGRIS", \
               "ERA5", \
               "MERRA2"/)
    end if

  colors = (/"blue","deepskyblue","green4","green","darkorchid4","magenta","black","grey30"/)

;******************************************************

  ;clmframes = (/-5.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,5./)
  clmframes = (/-3.,-2.,-1.,0.,1.,2.,3./)
  ;camframes = (/-1.,-0.75,-0.50,-0.25,0.,0.25,0.50,0.75,1./)
  camframes = (/-2.,-1.75,-1.5,-1.25,-1.,-0.75,-0.50,-0.25,0.,0.25,0.50,0.75,1.,1.25,1.5,1.75,2./)
  ;camframes = (/-5.,-4.75,-4.5,-4.25,-4.,-3.75,-3.5,-3.25,-3.,-2.75,-2.5,-2.25,-2.,-1.75,-1.5,-1.25,-1.,-0.75,-0.50,-0.25,0.,0.25,0.50,0.75,1.,1.25,1.5,1.75,2.,2.25,2.5,2.75,3.,3.25,3.5,3.75,4.,4.25,4.5,4.75,5./)

  dtclm = clmframes(1)-clmframes(0)
  dtcam = camframes(1)-camframes(0)

  xtime1 = clmframes
  xtime2 = camframes
  xtime3 = clmframes
  xtime4 = camframes


  if (melt) then
     if (inttop) then
        yaxis1 = "Cumulative Melt (mm)"
     else
        yaxis1 = "Melt (mm/day)"
     end if
  else
     if (inttop) then
       yaxis1 = "Cumulative Precip (mm)"
     else
       yaxis1 = "Precip (mm/day)"
      end if  
   end if

  ymin1 =  0.
  ymax1 =  150.
  
  ymin2 =  0.
  ymax2 =  150.

   if (melt) then
     if (inttop) then
        yaxis3 = "Area integrated Cumulative Melt (Gt)"
     else
        yaxis3 = "Area Integrated Melt (Gt/day)"
     end if
   else
     if (inttop) then
       yaxis3 = "Area Integrated Cumulative Precip (Gt)"
     else
       yaxis3 = "Area Integrated Precip (Gt/day)"
      end if
   end if

  ymin3 =  0.
  ymax3 =  2.4
  
  ymin4 =  0.
  ymax4 =  3.2
  
  xaxis = "Days relative to max overlap with GrIS"
  
  xmina = -3.25
  xmaxa =  3.25
  
  xminb = -2.25
  xmaxb =  2.25

  pltNam = "temp_precip"

;******************************************************
; create plot
;******************************************************
  wkstype = "pdf"
  wks = gsn_open_wks(wkstype,pltNam)               ; open workstation

  plot = new( (/2/), "graphic")

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

   res@tiXAxisFontHeightF = 0.015
   res@tiYAxisFontHeightF = 0.015
   res@tmXBLabelFontHeightF = 0.011
   res@tmYLLabelFontHeightF = 0.015
   
   res@gsnYRefLine = 0.
   
   ;;; transparent polygons
   gsres                = True
   gsres@gsFillOpacityF = 0.2;;0.3
   gsres@tfPolyDrawOrder = "PreDraw"
 

;******************************************************
; begin loop
;******************************************************
     
  do n = 0,nfiles-1
   
   if n.eq.(6) then
     if (melt) then
     else       
       print("load:"+subdir_scrip+fname11(n))
       f = addfile(subdir_scrip + fname11(n),"r")
       var2 = f->av
       xp2 = f->xp
       yp2 = f->yp
       delete(f)

       print("load:"+subdir_scrip+fname12(n))
       f = addfile(subdir_scrip + fname12(n),"r")
       var4 = f->av
       xp4 = f->xp
       yp4 = f->yp
       delete(f)
     
     end if
    else 
       print("load:"+subdir_scrip+fname11(n))
       f = addfile(subdir_scrip + fname11(n),"r")
       var2 = f->av
       xp2 = f->xp
       yp2 = f->yp
       delete(f)

       print("load:"+subdir_scrip+fname12(n))
       f = addfile(subdir_scrip + fname12(n),"r")
       var4 = f->av
       xp4 = f->xp
       yp4 = f->yp
       delete(f) 
   end if

   if (inttop) then
     if (n.eq.6 .and. (melt)) then
     else
       camcum = var2
       camcum = 0.
       ;camcum(0) = var2(0)
       n2 = dimsizes(camcum)

       up = yp2(1:n2)
       tmp= yp2(n2+1:2*n2)
       dn = tmp(::-1)
       delete(tmp)

       cumup2 = camcum
       cumup2 = 0.
       cumdn2 = cumup2
       ;cumup2(0) = (/up(0)/)
       ;cumdn2(0) = (/dn(0)/)
       do m = 1,n2-1
         camcum(m) = camcum(m-1) + dtcam*var2(m)
         cumup2(m) = cumup2(m-1) + dtcam*up(m)
         cumdn2(m) = cumdn2(m-1) + dtcam*dn(m)
       end do
       delete([/var2,up,dn/])
       var2 = camcum
       delete(camcum)
     end if
   end if

   if (intbot) then
     if (n.eq.6 .and. (melt)) then
     else
       camcum = var4
       camcum = 0.
       ;camcum(0) = var4(0)
       n4 = dimsizes(camcum)

       up = yp4(1:n4)
       tmp= yp4(n4+1:2*n4)
       dn = tmp(::-1)
       delete(tmp)
     
       cumup4 = camcum
       cumup4 = 0.
       cumdn4 = cumup4
       ;cumup4(0) = (/up(0)/)
       ;cumdn4(0) = (/dn(0)/)
       do m = 1,n4-1
         camcum(m) = camcum(m-1) + dtcam*var4(m)
         cumup4(m) = cumup4(m-1) + dtcam*up(m)
         cumdn4(m) = cumdn4(m-1) + dtcam*dn(m)
       end do
       delete([/var4,up,dn/])
       var4 = camcum
       delete(camcum)
    end if
  end if

   res@xyDashPatterns = 0
   res@xyLineThicknessF  = 4
   res@xyLineColors = colors(n)

   ;print(xtime2)
   ;print(var2)

   if (n.eq.0) then
     res@tiYAxisString   = yaxis1
     res@trYMinF         = ymin2
     res@trYMaxF         = ymax2
     res@trXMinF         = xminb
     res@trXMaxF         = xmaxb
     plot(0)  = gsn_csm_xy(wks,xtime2,var2,res)

     res@tiXAxisString = xaxis
     res@tiYAxisString   = yaxis3
     res@trYMinF         = ymin4
     res@trYMaxF         = ymax4
     res@trXMinF         = xminb
     res@trXMaxF         = xmaxb
     plot(1)  = gsn_csm_xy(wks,xtime4,var4,res)
     ;should be 3

     delete(res@tiXAxisString)

   else
     if (n.eq.6 .and. (melt)) then
     else
 
       tmp2 = gsn_csm_xy(wks,xtime2,var2,res)
       overlay(plot(0),tmp2) ;should be 1
       delete(tmp2)

       tmp4 = gsn_csm_xy(wks,xtime4,var4,res)
       overlay(plot(1),tmp4) ;should be 3
       delete(tmp4)
     end if   
   end if
   
   if (n.eq.6 .and. (melt)) then 
   else  
     delete([/var2,var4/])
   end if
   ;;now plot polygon shading
    
   if (n.eq.6 .and. (melt)) then
   else
     xp2 = xp2 - 1
     m2 = dimsizes(xp2)
     xxp2 = new( (/m2/), "float")
     do m = 0,m2-1
       xxp2(m) = camframes(tointeger(xp2(m)))
     end do
   end if

   if (inttop) then
     if (n.eq.6 .and. (melt)) then
     else
       yyp2 = new( (/m2/), "double")
       yyp2(0) = cumup2(0)
       yyp2(1:n2) = cumup2
       yyp2(n2+1:2*n2) = cumdn2(::-1)
     end if
   else
     if (n.eq.6 .and. (melt)) then
     else
       yyp2 = yp2
     end if
   end if

   gsres@gsFillColor    = colors(n)

   polystr = unique_string("hum")
;if (n.eq.2 .or. n.eq.nfiles-1) then
   if (n.eq.6 .and. (melt)) then
   else
     plot@$polystr$ = gsn_add_polygon(wks,plot(0),xxp2,yyp2,gsres) ;should be 1
   end if
 ;end if
   if (n.eq.6 .and. (melt)) then
   else
     delete([/yyp2,xp2,yp2/])
   end if

   m4 = m2
   if (n.eq.6 .and. (melt)) then
   else
     xxp4 = xxp2
  end if 
  if (n.eq.6 .and. (melt)) then
  else
    delete([/xxp2/])
  end if

   if (intbot) then
     if (n.eq.6 .and. (melt)) then
     else
       yyp4 = new( (/m4/), "double")
       yyp4(0) = cumup4(0)
       yyp4(1:n4) = cumup4
       yyp4(n4+1:2*n4) = cumdn4(::-1)
     end if
   else
     if (n.eq.6 .and. (melt)) then
     else
       yyp4 = yp4
     end if
   end if

   polystr = unique_string("hum")
;if (n.eq.2 .or. n.eq.nfiles-1) then
   if (n.eq.6 .and. (melt)) then
   else
     plot@$polystr$ = gsn_add_polygon(wks,plot(1),xxp4,yyp4,gsres) ;should be 3
   end if
;end if
if (n.eq.6 .and. (melt)) then
else
  delete([/xxp4,yyp4,xp4,yp4/])
end if
 end do;;n


;*****************************************************
; Add legend
;*****************************************************
 txres = True
 txres@txFontHeightF = 0.010
 txres@txJust = "CenterLeft"

 resln             = True
 resln@gsLineThicknessF = res@xyLineThicknessF

 yy = (/0.74,0.725,0.71,0.695,0.68,0.665,0.65,0.635/)+0.1;-0.07
 xx = 0.17
 dx = 0.03
 do n = 0,nfiles-1
   resln@gsLineColor = colors(n)
   gsn_polyline_ndc(wks,(/xx,xx+dx/),(/yy(n),yy(n)/),resln)
   gsn_text_ndc(wks,"  " + labels(n),xx+dx,yy(n),txres)
 end do

 txres@txFontHeightF = 0.015
 txres@txJust = "CenterCenter"

 delete(yy)
 yy = 0.71
 xx = 0.30
 ;if (melt) then
 ;  gsn_text_ndc(wks,"Snowmelt",xx,yy,txres)
 ;else
 ;  gsn_text_ndc(wks,"CLM (daily mean)",xx,yy,txres)
 ;end if
 
 xx = 0.75
 
;if (melt) then
;  gsn_text_ndc(wks,"Snowmelt + Icemelt",xx,yy,txres)
;else
;  gsn_text_ndc(wks,"CAM (6-hourly instantaneous)",xx,yy,txres)
;end if
;;drawNDCGrid(wks)
;*****************************************************
; Plot panel
;*****************************************************

  resP                  = True                   ; modify the panel plot
  resP@gsnFrame         = False                  ; don't advance panel plot

  ;;resP@txString = plotTitle
  ;;resP@gsnPanelFigureStrings = runnams
  resP@gsnPanelFigureStrings= (/"a.","b."/)
  resP@gsnPanelFigureStringsFontHeightF = 0.03
  resP@gsnPanelFigureStringsPerimOn     = True
  resP@gsnPanelFigureStringsBackgroundFillColor = "white"
  resP@amJust           = "BottomLeft"
  resP@gsnPanelLeft = 0.05
  resP@gsnPanelRight = 0.95
  ;;resP@gsnPanelBottom   = 0.1

  gsn_panel(wks,plot,(/2,1/),resP)

frame(wks)

end
