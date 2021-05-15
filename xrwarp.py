import rasterio
from rasterio.warp import calculate_default_transform
from rasterio.transform import array_bounds

from osgeo import gdal
import xarray as xr
import numpy as np
import itertools
from operator import itemgetter


def get_transform(ds, x_coords, y_coords):
    x_size, y_size = ds.dims[x_coords], ds.dims[y_coords]
    i_top, i_bottom = [ds[y_coords].values[y] for y in [0, -1]]
    i_left, i_right = [ds[x_coords].values[x] for x in [0, -1]] 
    res_x = (i_right - i_left) / (x_size - 1)
    res_y = (i_bottom - i_top) / (y_size - 1)
    north = i_top - (res_y/2.0)
    west = i_left - (res_x/2.0)
    east = i_right + (res_x/2.0)
    south = i_bottom + (res_y/2.0)
    transform = rasterio.transform.from_bounds(west, south, east, north, x_size, y_size)
    return transform


# warp is done by moving values from each time dimension to a band
# src dataset
#
def create_mem_src(x_size, y_size, transform, crs):
    src_ds = gdal.GetDriverByName('MEM').Create(r"test", 
                                                xsize=x_size, 
                                                ysize=y_size, 
                                                bands=1, 
                                                eType=gdal.GDT_Float64)
    src_ds.SetGeoTransform(transform.to_gdal())
    src_ds.SetProjection(crs.to_wkt())
    tb = src_ds.GetRasterBand(1)
    tb.SetNoDataValue(np.nan)
    tb = None
    return src_ds


def gen_get_coords(affine):
    # generate convenience function to co-ordinates using affine and cell_size
    # move the cell_size by 1/2 to represent the center
    # can be done more elegantly?
    def _f(x): 
        return list(x + 0.5) * affine
    return _f

def fill_nans_with_gdal(data, transform, crs):
    y_size, x_size = data.shape
    with rasterio.Env(GDAL_CACHEMAX=512) as env:
        dst_ds = gdal.GetDriverByName('MEM').Create(r"test",xsize=x_size, ysize=y_size, bands=1, eType=gdal.GDT_Float64)
        dst_ds.SetGeoTransform(transform.to_gdal())
        dst_ds.SetProjection(crs.to_wkt())
        dst_ds.GetRasterBand(1).WriteArray(data)
        tb = dst_ds.GetRasterBand(1)
        tb.SetNoDataValue(np.nan)
        gdal.FillNodata(tb, maskBand = None, maxSearchDist = 6, smoothingIterations = 0)
        tb = None
        res = dst_ds.GetRasterBand(1).ReadAsArray()
    return res


def return_coords(affine, width, height):
    # get new long and lat co-ords
    import itertools
    from operator import itemgetter
    get_coord = gen_get_coords(affine)

    new_xs = np.stack((np.arange(width), np.repeat(0,width)), 1)
    new_xs = np.apply_along_axis(get_coord, 1, new_xs)[...,0]
    
    new_ys = np.stack((np.repeat(0, height), np.arange(height)), 1)
    new_ys = np.apply_along_axis(get_coord, 1, new_ys)[...,1]
    
    return new_xs, new_ys


def warp_ds(ds, 
            src_crs, 
            new_cell_size=None, 
            dst_crs=None, 
            var_name='rain', 
            xname='longitude', 
            yname='latitude', 
            tname='time', 
            resampling_alg=gdal.GRA_CubicSpline):
    from tqdm import tqdm
    # todo: not sure what happens if the order of co-ordinates in the source xarray
    # does not match the order expected here (time, latitude, longitude)
    x_size, y_size = ds.dims[xname], ds.dims[yname]
    if not new_cell_size:
        new_cell_size = (x_size, y_size)
    if not dst_crs:
        dst_crs= src_crs
    time_size = ds.dims[tname]

    ds_src_transform = get_transform(ds, x_coords=xname, y_coords=yname)
    west, south, east, north = array_bounds(height=y_size, 
                                            width=x_size, 
                                            transform=ds_src_transform)
    src_ds = create_mem_src(x_size, y_size, ds_src_transform, src_crs)
    dst_transform, dst_width, dst_height = calculate_default_transform(src_crs=src_crs, 
                                                                       dst_crs=dst_crs, 
                                                                       width=x_size, height=y_size,
                                                                       left=west, bottom=south, right=east, top=north,
                                                                       resolution=(new_cell_size, new_cell_size))
    out_bounds = array_bounds(height=dst_height, width=dst_width, transform=dst_transform)
    
    wrapopts = gdal.WarpOptions(xRes=new_cell_size, 
                            yRes=new_cell_size, 
                            srcSRS=src_crs.to_wkt(),
                            dstSRS=dst_crs.to_wkt(),
                            outputBounds=out_bounds,
                            resampleAlg=resampling_alg,
                            dstNodata=np.nan)
    new_xs, new_ys = return_coords(dst_transform, dst_width, dst_height)
    time_values = times = ds.indexes[tname]
    warped_data = np.zeros((time_size, dst_height, dst_width))
    for ti, tvalue in tqdm(list(enumerate(ds.indexes[tname]))):
        data = ds[var_name][ti,...].values.copy()
        src_ds.GetRasterBand(1).WriteArray(data)
        tb = src_ds.GetRasterBand(1)
        gdal.FillNodata(tb, maskBand = None, maxSearchDist = 6, smoothingIterations = 0)
        _ = tb.ReadAsArray()   
        warp_ras = gdal.Warp(r"/vsimem/wrap_singletimestamp.tiff", src_ds, options=wrapopts)
        warped_slice = warp_ras.GetRasterBand(1).ReadAsArray().copy()
        if warped_slice.shape != (dst_height, dst_width):
            raise ValueError(warped_slice.shape, (dst_height, dst_width))
        warped_data[ti,...] = warped_slice
        del warp_ras
    warped_ds = xr.DataArray(warped_data, 
                         coords=[times, new_ys, new_xs],
                         dims=[tname, yname, xname])
    return warped_ds