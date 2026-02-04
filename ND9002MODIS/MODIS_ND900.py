
# quick subroutine to analyze the W data
import sys
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
#import netCDF4 as nc
import os
import glob
import xarray as xr
import time
from dask.distributed import Client, LocalCluster
from datetime import datetime, timedelta
#from scipy import interpolate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchsummary import summary
import json
import gc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import xesmf as xe
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.gridspec as gridspec

#M = np.array([3, 5e-5, 0.5, 1.5, 5e-4, 0.5, 300e-6, 1000, 1., 800, 50, 1, 1])
m= [49.5, 27.9, 307.2, 0.5, 0.06, 0.02, 10., 50., 223.] 
s = [37.1, 15.9, 16.1, 1, 0.06, 0.02, 5., 20., 10.]
#['swcf', 'lwcf', 'T2M', 'CLDTT', 'LWP', 'IWP', 'MDSCLDSZWTR',  'MDSCLDSZICE', 'TROPT']

np.set_printoptions(precision=3)

def nb(x, prec=1):
    	return np.format_float_positional(x, precision=prec)
 
def amean(ds):
    weights = np.cos(np.deg2rad(ds.lat))
    #weights = np.cos(0.01745329*ds.lat)
    ds_weighted = ds.weighted(weights)
    return ds_weighted.mean(("lon", "lat"),  skipna=True)
    
def corrcoefxy( x, y ):    
    mean_x = np.mean( x )
    mean_y = np.mean( y )
    std_x  = np.std ( x )
    std_y  = np.std ( y )
    n      = len    ( x )
    return (( x - mean_x ) * ( y - mean_y )).sum() / n / ( std_x * std_y )
    
def stand_by_cell(ds, invert = False, var =  'N', lv=[]):
	#assume we have std and mean saved in the same folder
    pth =  '../stats/' + var + '_stats.nc4'
    dst  =  xr.open_mfdataset(pth, parallel=True)
    if lv:
    	ds_mean =  dst[var+'_mean'].sel(lev=lv)
    	ds_std =   dst[var+'_std'].sel(lev=lv)
    else:
    	ds_mean =  dst[var+'_mean']
    	ds_std =   dst[var+'_std']


    ds_std = ds_std.where(ds_std > 0.)
    if not invert:        
        ds = (ds-ds_mean)/ds_std
    else:
    	ds =  ds*ds_std + ds_mean
    return ds


#def power_scaler(ds, nexp):
#   func  = lambda x, n: np.power(np.where(x<0, 0, x), n)  
#   return xr.apply_ufunc(func, ds, nexp, dask='parallelized') 

def linear_scaler(ds, cst):
   func  = lambda x, n: x*n 
   return xr.apply_ufunc(func, ds, cst, dask='parallelized') 
 

def calcRF (ds):
    ds['net'] = ds.SWTNET -  ds.OLR
    ds['swcf'] =  ds.SWTNETC -  ds.SWTNET
    ds['lwcf'] = ds.OLRC -  ds.OLR 
    #ds2 = ds['net', 'swcf', 'lwcf'] 
    #print(ds2)
    return ds #ds[['net', 'swcf', 'lwcf']] 



def get_data(var_out = 'N', pth_out = '', var_in = 'XX', pth_in = '', lv = 900):

    chk  = {"time": 1}
    print(pth_in)
    Nout  =  xr.open_mfdataset(pth_out, parallel=True, chunks=chk)[var_out]       
    Nin  =  xr.open_mfdataset(pth_in, parallel=True, chunks=chk)[var_in].sel(lev=lv)

    #standardize
    #Nout =  Nout.where(Nout < 1e10)
    #Nin =  Nin.where(Nin < 1e10)
    Nout =  Nout.where(Nout > 0)
    Nin =  Nin.where(Nin > 0)
    
    Nin = stand_by_cell(Nin, var =  var_in, lv=  lv)
    Nout = stand_by_cell(Nout, var =  var_out)
       
   
    regridder = xe.Regridder(Nin, Nout, 'bilinear', periodic=True) #make sure they are exactly the same grid 
    Nin =  regridder(Nin)
    
    # Align time dimension
    X_time = Nin['time'].dt.floor('D')
    Y_time = Nout['time'].dt.floor('D')

    shared_time = np.intersect1d(X_time.values, Y_time.values)
    Nin = Nin.sel(time=np.isin(X_time, shared_time))
    Nout = Nout.sel(time=np.isin(Y_time, shared_time))
    
    assert np.array_equal(
    Nin['time'].dt.floor('D').values,
    Nout['time'].dt.floor('D').values
		    ), "Time dimensions are not aligned after filtering"


    Nin =  Nin.squeeze()
    return Nin.transpose('time', 'lat', 'lon').fillna(0).load() , Nout.transpose('time', 'lat', 'lon').load() 


########### U-NET
class MaskedLatitudeWeightedMSELoss(nn.Module):
    def __init__(self, latitudes):
        super().__init__()
        lat_radians = torch.deg2rad(latitudes)
        self.register_buffer('lat_weights', torch.cos(lat_radians).view(1, 1, -1, 1))  # [1, 1, H, 1]

    def forward(self, pred, target):
        mask = ~torch.isnan(target)

        # Broadcast latitude weights to match [B, 1, H, W]
        weights = self.lat_weights.to(pred.device).expand_as(pred)

        # Apply the mask
        pred = pred[mask]
        target = target[mask]
        weights = weights[mask]

        loss = ((pred - target) ** 2) * weights
        return loss.sum() / (weights.sum().clamp(min=1e-8))

class MaskedNaNMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Create a mask: 1 where target is not NaN, 0 where NaN
        mask = ~torch.isnan(target)
        
        # Ensure prediction and target match in shape
        pred = pred[mask]
        target = target[mask]

        # Compute MSE only on valid (non-NaN) elements
        return torch.mean((pred - target) ** 2)

class MaskedNaNL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        mask = ~torch.isnan(target)
        pred = pred[mask]
        target = target[mask]
        return torch.mean(torch.abs(pred - target))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        alpha =  0.05
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_ch),
            #nn.ReLU(inplace=True),
            #nn.PReLU(num_parameters=out_ch),
            nn.LeakyReLU(negative_slope=alpha, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_ch),
            #nn.ReLU(inplace=True),
            #nn.PReLU(num_parameters=out_ch),
            nn.LeakyReLU(negative_slope=alpha, inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.conv(x)

def match_tensor(source, target):
    """Resize source tensor spatially to match target using interpolation."""
    if source.shape[2:] != target.shape[2:]:
        source = F.interpolate(source, size=target.shape[2:], mode='bilinear', align_corners=False)
    return source

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout=0.01):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = ConvBlock(in_channels, 32, dropout)
        self.enc2 = ConvBlock(32, 64, dropout)
        self.enc3 = ConvBlock(64, 128, dropout)
        self.enc4 = ConvBlock(128, 256, dropout)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 512, dropout)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256, dropout)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128, dropout)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64, dropout)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32, dropout)

        # Output
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1, bias=True)

    def forward(self, x_in):
        x1 = self.enc1(x_in)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        x = self.bottleneck(self.pool(x4))

        x = self.up4(x)
        x4 = match_tensor(x4, x)
        x = self.dec4(torch.cat([x, x4], dim=1))

        x = self.up3(x)
        x3 = match_tensor(x3, x)
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x2 = match_tensor(x2, x)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x1 = match_tensor(x1, x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        x = self.out_conv(x)

        # Ensure exact match with input shape
        x = F.interpolate(x, size=x_in.shape[2:], mode='bilinear', align_corners=False)
        return x




def shift_lon(lon):
    return xr.where(lon > 180., lon - 360., lon)
    


# Custom dataset class for xarray data
class XarrayDataset(Dataset):
    def __init__(self, xarray_data, xarray_labels):
        self.data = xarray_data       
        self.labels = xarray_labels

    def __len__(self):
    	return len(self.data)

    def __getitem__(self, idx):
        # Convert xarray data to torch tensors
        data_tensor = torch.tensor(self.data[idx].values, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(self.labels[idx].values, dtype=torch.float32).unsqueeze(0)
        return data_tensor, label_tensor

#=========Training===================
def training_val(model, train_loader, val_loader, hp, device, checkpoint=None, latitudes=[]):

    # Early stopping criteria
    min_rolling_val_loss = float('inf')
    patience_counter = 0
    # Lists to store loss values
    train_losses = []
    val_losses = []
    Nr = hp.get('rolling_Nr', 5)  # Number of epochs for rolling average

    # Define loss function and optimizer
    #criterion = nn.MSELoss()
    #criterion = MaskedNaNMSELoss().to(device)
    criterion = MaskedLatitudeWeightedMSELoss(latitudes).to(device) 
    optimizer = optim.AdamW(model.parameters(), lr=hp['l_r'], weight_decay=1e-5)
    num_epochs =  hp['nepochs']
    patience =  hp['patience']
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
 
    #load checkpoint
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        
    
    
           # Training loop

    for epoch in range(hp['nepochs']):
        # Training phase
        start_time = time.time()  # Start timing the epoch
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            # Zero the parameter gradients
            #print("Input shape:", inputs.shape)
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the device
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            
            #print("----inputs:", inputs.shape)
            #print("----targets:", targets.shape)
            #print("----outputs:", outputs.shape)

            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # Move data to the device
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Compute rolling average of the last Nr validation losses
        if len(val_losses) >= Nr:
            rolling_avg_val_loss = sum(val_losses[-Nr:]) / Nr
        else:
            rolling_avg_val_loss = sum(val_losses) / len(val_losses)
            
        scheduler.step(rolling_avg_val_loss)
        #Time per epoch
        epoch_time = time.time() - start_time
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f} s, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Current LR: {optimizer.param_groups[0]["lr"]}')
        # Early stopping
        if rolling_avg_val_loss < min_rolling_val_loss:
            min_rolling_val_loss = rolling_avg_val_loss
            patience_counter = 0
            # Save the best model
            
            print('model saved')
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_losses': val_losses,
            }, hp['model_name'] + '.pth')
            #torch.save(model.state_dict(), )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping')
                break

    print('Finished Training')

    return train_losses, val_losses


if __name__ == '__main__':
   #client = Client()
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Check for multiple GPUs
    
    #prefix = '2001'
    prefix= '200'    
    pth_out= f"../data/MODIS_COSP_GR18/mod_ND_Gr18_" 
    #pth_out= f"../MODIS_ND/Grosvenor2018/daily_ND/mod_ND_Gr18_"
     
    pth_in= f"../data/aci_tavg_1dy_glo_L360x181_p27/GiOCEAN_e1.aci_tavg_1dy_glo_L360x181_p27.tavg1D."
    #pth_in = f"/gpfsm/dnb07/projects/p281/GiOcean_data/dailies/aci_tavg_1dy_glo_L360x181_p27/GiOCEAN_e1.aci_tavg_1dy_glo_L360x181_p27.tavg1D."
    var_in = 'NCPL_VOL' 
    var_out = 'MOD_TAU5_TTOP268'  
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device===', device)
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs" if num_gpus > 1 else "Using a single GPU or CPU")

   
    hp = {
        'nepochs': 5000,
        'batch_sz': 8,
        'l_r': 0.0004,
        'patience': 20,
        'model_name' : "N9002Nmodis",
        'rolling_Nr': 10         
     }
    
    do_train = False
    do_test = True
	
    if do_train: 

       
        X, Y  = get_data(var_out = var_out, pth_out = pth_out + f'{prefix}*.nc4' , var_in = var_in, pth_in = pth_in + f'{prefix}*.nc4')
        assert not np.any(np.isnan(X))        
        print('---X---', X)
        print('---Y---', Y)
        #assert not np.any(np.isnan(Y))
         
        Xs = []
        Ys =  []
     
        Xs =  X.values.shape
        print('---X---', Xs)
        print('---Y---', Y.values.shape)
        dataset = XarrayDataset(X, Y)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=hp['batch_sz'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hp['batch_sz'], shuffle=False)

        UNN = UNet(in_channels=1, out_channels=1)
        if os.path.exists(hp['model_name'] + '.pth'):
            # Load model:
            print('-----Checkpoint exists! Restarting training')
            checkpoint = torch.load(hp['model_name'] + '.pth')
            UNN.load_state_dict(checkpoint['model_state_dict'])
        	 
        # Wrap the model with DataParallel if more than one GPU is available
        if num_gpus > 1:
        	UNN = nn.DataParallel(UNN)

        UNN.to(device)  # Move the model to the appropriate device
 
        #summary(NN(input_size(Xs[3], Xs[1], Xs[2], Yall.shape[1])))
        latitudes = torch.tensor(Y['lat'].values)
        train_losses, val_losses =  training_val(UNN, train_loader, val_loader, hp, device, latitudes=latitudes)

        #print(train_losses)             
                     #plot loss
        plt.switch_backend('agg')
        plt.plot(train_losses)
        plt.plot(val_losses)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.xlim(left=1) #xmin is your value
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(hp['model_name'] + '_loss.png')
        


    if do_test:
        # === Load Data ===
        prefix = '2011'
        #pr= '200901'
        X, Y_true  = get_data(var_out = var_out, pth_out = pth_out + f'{prefix}*.nc4' , var_in = var_in, pth_in = pth_in + f'{prefix}*.nc4')  
        # Y_true is the target xarray, shape [sample, lat, lon]
        print('---X---', X)
        print('---Y---', Y_true)
        Ys =  Y_true.data.shape
         

        # Dummy Y for Dataset, not used in inference
        dataset = XarrayDataset(X, Y_true)
        test_loader = DataLoader(dataset, batch_size=hp['batch_sz'], shuffle=False)

        # === Load Model ===
        UNN = UNet(in_channels=1, out_channels=1)
        if num_gpus > 1:
            UNN = nn.DataParallel(UNN)
        UNN.to(device)

        if os.path.exists(hp['model_name'] + '.pth'):
            print('-----Checkpoint exists! Loading weights')
            checkpoint = torch.load(hp['model_name'] + '.pth', map_location=device)
            UNN.load_state_dict(checkpoint['model_state_dict'])
            summary(UNN, input_size=(1, Ys[1], Ys[2]))
            UNN.eval()

        # === Inference ===
        all_predictions = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = UNN(inputs)
                all_predictions.append(outputs.cpu())

        all_predictions = torch.cat(all_predictions, dim=0)  # [N, 1, lat, lon]
        preds = all_predictions.squeeze(1).numpy()  # [N, lat, lon]
        


        # === Create xarray prediction ===
        pred_xr = xr.DataArray(
            preds,
            dims=Y_true.dims,
            coords=Y_true.coords,
            name='prediction'
        )

        
        pred_xr = pred_xr.where(pred_xr >= 0)
        Y_true = Y_true.where(Y_true >= 0) 
        pred_xr = pred_xr.where(Y_true >= 0)
        
        #de-standardazie 
        
        pred_xr = stand_by_cell(pred_xr, var =  var_out, invert =  True)  
        Y_true = stand_by_cell(Y_true, var =  var_out, invert =  True)
        X_in = stand_by_cell(X, var =  var_in, invert =  True, lv=  900 )*1e-6
        
 
        Npre = nb(amean(pred_xr).mean().load())
        Ntrue = nb(amean(Y_true).mean().load())
        Nin = nb(amean(X_in).mean().load()) 
 
        # === Compute Averages, Differences, and RMSE ===
        avg_in = X_in.mean(dim='time')
        avg_pred = pred_xr.mean(dim='time')
        avg_true = Y_true.mean(dim='time')
        avg_diff = 100 * (pred_xr - Y_true) / Y_true
        avg_diff = avg_diff.mean(dim='time')
        #avg_diff = 100 * (avg_pred - avg_true) / avg_true 
        
        df = nb(amean(avg_diff).mean().load())
        rmse = np.sqrt(((pred_xr - Y_true) ** 2))/Y_true
        rmse=rmse.mean(dim='time')
        rm = nb(amean(rmse).mean().load(), prec=2)

        # === Setup for plotting ===

        # Global font settings
        plt.rcParams.update({
            'font.size': 8,
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8
        })


        # === Setup ===
        Nmax = 1000
        Nmin = 1
        sz = 8
        lbsz = 8
        proj = ccrs.Robinson()
        log_norm = mcolors.LogNorm(vmin=Nmin, vmax=Nmax)

       	fig = plt.figure(figsize=(7.2, 4.3))
        gs = gridspec.GridSpec(2, 6, figure=fig)  # 6-column layout

        # --- Top row: evenly across 6 columns ---
        axes = [
            fig.add_subplot(gs[0, 0:2], projection=proj),  # 1. Input
            fig.add_subplot(gs[0, 2:4], projection=proj),  # 2. Predicted
            fig.add_subplot(gs[0, 4:6], projection=proj)   # 3. True
        ]

        # --- Bottom row: centered using columns 1:3 and 3:5 ---
        axes.append(fig.add_subplot(gs[1, 1:3], projection=proj))  # 4. Percent Diff
        axes.append(fig.add_subplot(gs[1, 3:5], projection=proj))  # 5. RMSE

        # === Plotting ===

        # 1. Input
        im0 = avg_in.plot.pcolormesh(ax=axes[0], transform=ccrs.PlateCarree(), cmap='turbo',
                                     norm=log_norm, add_colorbar=False)
        axes[0].set_title('(a)', loc =  'left')
        axes[0].set_title( Nin + r' cm$^{-3}$', loc = 'center')
        axes[0].coastlines()
        axes[0].add_feature(cfeature.BORDERS, linewidth=0.5)
        cb0 = fig.colorbar(im0, ax=axes[0], orientation='horizontal', pad=0.05, shrink=0.7, extend='both')
        cb0.ax.tick_params(labelsize=sz)
        cb0.set_label(r'Nd_900 (cm$^{-3}$)', fontsize=lbsz)
        
        
        # 2. Predicted
        im1 = avg_pred.plot.pcolormesh(ax=axes[1], transform=ccrs.PlateCarree(), cmap='turbo',
                                       norm=log_norm, add_colorbar=False)
        axes[1].set_title('(b)', loc =  'left')
        axes[1].set_title( Npre + r' cm$^{-3}$', loc = 'center')
        axes[1].coastlines()
        axes[1].add_feature(cfeature.BORDERS, linewidth=0.5)
        cb1 = fig.colorbar(im1, ax=axes[1], orientation='horizontal', pad=0.05, shrink=0.7, extend='both')
        cb1.ax.tick_params(labelsize=sz)
        cb1.set_label(r'Pred. Nd (cm$^{-3}$)', fontsize=lbsz)

        # 3. Truth
        im2 = avg_true.plot.pcolormesh(ax=axes[2], transform=ccrs.PlateCarree(), cmap='turbo',
                                       norm=log_norm, add_colorbar=False)
        axes[2].set_title('(c)', loc =  'left')
        axes[2].set_title( Ntrue + r' cm$^{-3}$', loc = 'center')       
        axes[2].coastlines()
        axes[2].add_feature(cfeature.BORDERS, linewidth=0.5)
        cb2 = fig.colorbar(im2, ax=axes[2], orientation='horizontal', pad=0.05, shrink=0.7, extend='both')
        cb2.ax.tick_params(labelsize=sz)
        cb2.set_label(r'COSP Nd (cm$^{-3}$)', fontsize=lbsz)

        # 4. Percent Difference
        im3 = avg_diff.plot.pcolormesh(ax=axes[3], transform=ccrs.PlateCarree(), cmap='RdBu_r',
                                       vmin=-50, vmax=50, add_colorbar=False)
         
        axes[3].set_title('(d)', loc =  'left')
        axes[3].set_title(f'{df} %'  , loc = 'center')      
        axes[3].coastlines()
        axes[3].add_feature(cfeature.BORDERS, linewidth=0.5)
        cb3 = fig.colorbar(im3, ax=axes[3], orientation='horizontal', pad=0.05, shrink=0.7, extend='both')
        cb3.ax.tick_params(labelsize=sz)
        cb3.set_label(f'Avg % Difference', fontsize=lbsz)

        # 5. RMSE
        im4 = rmse.plot.pcolormesh(ax=axes[4], transform=ccrs.PlateCarree(), cmap='YlOrRd',
                                   vmin=0, vmax=1, add_colorbar=False)

        axes[4].set_title('(e)', loc =  'left')
        axes[4].set_title(rm , loc = 'center')         
        axes[4].coastlines()
        axes[4].add_feature(cfeature.BORDERS, linewidth=0.5)
        cb4 = fig.colorbar(im4, ax=axes[4], orientation='horizontal', pad=0.05, shrink=0.7, extend='max')
        cb4.ax.tick_params(labelsize=sz)
        cb4.set_label(f'RMSE/Nd', fontsize=lbsz)

        #plt.tight_layout()
                # Replace tight_layout with fine-tuned spacing
        fig.subplots_adjust(
            top=0.95,
            bottom=0.05,
            left=0.05,
            right=0.98,
            wspace=0.15,
            hspace=0.15
        )

        plt.savefig(hp['model_name'] + '_test.png', dpi=300, bbox_inches='tight')
        print('------------done')
