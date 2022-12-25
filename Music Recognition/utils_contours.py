import cv2 as cv
import pickle
import pandas as pd
def select_contours(img, filled_value=100):
    # Mouse callback
    def select_contour(event, x, y, flags, param):
        # Return if clicked on black
        if img[y,x]==0:
            return
            
        # Fill or unfill contour
        if event == cv.EVENT_LBUTTONDOWN:
            fill_value = filled_value if img[y,x]==255 else 255
            cv.floodFill(img, None, (x,y), fill_value)

        # Update image
        cv.imshow('image', img)
        return


    # Main loop

    cv.namedWindow='image'
    cv.imshow('image', img)
    cv.setMouseCallback('image', select_contour)

    save_status = True
    while True:
        cv.imshow('image', img)
        k = cv.waitKey(20)
        if k==ord('s'):
            break
        elif k==ord('q'):
            save_status = False
            break

    cv.destroyAllWindows()
    return img, save_status

def get_contour_data(mask, line_sep, specified_contours=None, contour_df=None, is_selected=True, filters={}, return_contours=False):
    if contour_df is None:
        contour_df = pd.DataFrame(columns=['state', 'cx', 'cy','area','width','height','aspectratio','extent','solidity', 'normangle'])

    if specified_contours is None:
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    else:
        contours = specified_contours

    valid_contours = []
    for c in contours:
        # Skip if too simple
        if len(c)<5:
            continue
        
        # Find centroid
        M = cv.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        # Computations
        area = cv.contourArea(c)
        if area>5*line_sep**2:
            continue
        _,_,w,h = cv.boundingRect(c)
        hull = cv.convexHull(c)
        hull_area = cv.contourArea(hull)
        _,(MA,ma),angle = cv.fitEllipse(c)
        
        
        # Scale distances by line_sep
        area = area/line_sep**2
        hull_area = hull_area/line_sep**2
        MA = MA/line_sep
        ma = ma/line_sep
        w = w/line_sep
        h = h/line_sep
        
        # Ratios
        aspect_ratio = float(w)/h
        extent = float(area)/(w*h)
        solidity = float(area)/hull_area
        normangle = angle/180
        
        # Check filters
        if filters:
            if 'area' in filters.keys():
                min_area, max_area = filters['area']
                if area<min_area or area>max_area:
                    continue
            if 'solidity' in filters.keys():
                min_solidity, max_solidity = filters['solidity']
                if solidity<min_solidity or solidity>max_solidity:
                    continue

        # Append to dataframe
        contour_df.loc[len(contour_df)] = [is_selected, cx, cy, area, w, h, aspect_ratio, extent, solidity, normangle]
        valid_contours.append(c)

    if return_contours:
        return contour_df, valid_contours
    else:
        return contour_df
    