# **************************************************************************************
# LiquPy: Open-source Python Library for Soil Liquefaction and Lateral Spread Analysis
# https://github.com/LiquPy/LiquPy
# **************************************************************************************

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

class Borehole:
# borehole object

    number_of_holes = 0

    # customize visualization
    viz_liquefied_text_kwargs = {'color': (0, 0, 0, 0.4), 'horizontalalignment': 'center', 'verticalalignment': 'center'}
    viz_dashed_guidelines = {'color': (0, 0, 1, 0.05), 'ls': '--'}

    def __init__(self, bore_log_data, name=None, units='metric'):
        Borehole.number_of_holes += 1

        self.bore_log_data = bore_log_data
        self.name = name
        
        if units == 'metric':
            self.units_length = 'm'
            self.units_area = '$m^2$'
        elif units == 'british':
            self.units_length = 'ft'
            self.units_area = '$ft^2$'

    def __del__(self):
        Borehole.number_of_holes -= 1

    
    def simplified_liquefaction_triggering_fos(self, Pa, M, Zw=0, sampler_correction_factor=1,
                                               liner_correction_factor=1., hammer_energy=60, rod_extension=1,
                                               fs_threshold=1.):
        """ simplified liquefaction triggering analysis - stress-based

        Parameters
        ----------
        Pa : float
          Peak ground acceleration (g)

        M : float
          Earthquake magnitude

        Zw : float, default=0
          water table depth (in self.units_length units)

        sampler_correction_factor : float, default=1.

        liner_correction_factor : float, default=1.

        hammer_energy : float, default=60.

        rod_extension : float, default=1.

        output : 'fs' or 'probability', default='fs'
          determines the approach, deterministic or probabilistic
          
        rd_method : in ['Idriss1999', 'LiaoWhitman1986', 'Golesorkhi1989'], default= 'Idriss1999'
          Method of shear stress reduction factor
          
        fc_method : in ['BI2004', 'cetin2004'] , default= 'BI2004'
          Method of adjustments for fines content

        fs_threshold : float, default=1.
          Factor of safety threshold to consider soild as liqufied

        """
    
        self.Pa = Pa
        self.M = M
        self.Zw = Zw
        self.sampler_correction_factor = sampler_correction_factor
        self.liner_correction_factor = liner_correction_factor
        self.hammer_energy = hammer_energy
        self.rod_extension = rod_extension
        self.fs_threshold = fs_threshold

        output = []
        sigmavp = 0
        sigmav = 0
        depth = 0
        hydro_pore_pressure = 0
        gamma = self.bore_log_data.iloc[0, 6]
        for i, row in self.bore_log_data.iterrows():
            rod_length = row[1] + rod_extension
            Nspt = row[2]
            ce = hammer_energy / 60
            if rod_length < 3:
                cr = 0.75
            elif rod_length <= 4:
                cr = 0.8
            elif rod_length < 6:
                cr = 0.85
            elif rod_length <= 10:
                cr = 0.95
            else:
                cr = 1
            cs = sampler_correction_factor
            cb = liner_correction_factor
            N60 = Nspt * ce * cr * cs * cb

            sigmav = sigmav + (row[1] - depth)*(gamma)
            sigmavp = sigmavp + (row[1] - depth)*(gamma - 9.81)

            if row[4] == 1: # nonliquefiable
                N60 = 'n.a.'
                N160 = 'n.a.'
                N160cs = 'n.a.'
                MSF = 'n.a'
                k_sigma = 'n.a.'
                CN = 1
                delta_n = 1
            else:
                if sigmavp == 0:
                    CN = 1
                else:
                    CN = min(1.7, 2.2 / (1.2 + (sigmavp/100)))

                N160 = CN*N60  #  use of (N1)60 proposed by Liao and Whitman (1986)


                # Adjustments for fines content
                delta_n = np.exp(1.63 + 9.7/(row[5]+0.01) - (15.7/(row[5]+0.01))**2)
                N160cs = N160 + delta_n
                    
            # Shear stress reduction factor (depth in meters)
            if row[1] > 20: # 20 in meters
                warnings.warn('CSR (or equivalent rd values) at depths greater than about 20 m should be based on site response studies (Idriss and Boulanger, 2004)')

            if row[1] <= 34:
                rd = np.exp((-1.012-1.126*np.sin(row[1]/11.73+5.133)) + (0.106+0.118*np.sin(row[1]/11.28+5.142))*M)
            else:
                rd = 0.12*np.exp(0.22*M)

            # Magnitude scaling factor
            # Idriss (1999), default value
            MSF = min(1.8, 6.9 * np.exp(-M / 4) - 0.058)

            # Overburden correction factor
            # Boulanger and Idriss (2004)
            if N160cs <= 37:
                C_sigma = 1 / (18.9 - 2.55 * np.sqrt(N160cs))
            else:
                C_sigma = 0.3
                
            k_sigma = min(1.1, 1 - (C_sigma) * np.log(sigmavp / 100))

            # Earthquake-induced cyclic stress ratio (CSR)                
            CSR = 0.65*sigmav/sigmavp*Pa*rd*(1/(MSF*k_sigma))

            if row[4] == 1 or row[1] < Zw:
                CRR0 = 'n.a.'
                CRR = 'n.a.'
                FoS = 'n.a.'
                
            else:
                # SPT Triggering correlation of liquefaction in clean sands
                # Idriss and Boulanger (2004) & Idriss and Boulanger (2008)
                if N160cs < 37.5: # 37.5 in meters
                    CRR0 = np.exp(N160cs / 14.1 + (N160cs / 126) ** 2 - (N160cs / 23.6) ** 3 + (N160cs / 25.4) ** 4 - 2.8)
                else:
                    CRR0 = 2

                # Cyclic resistance ratio (CRR)
                CRR = min(2., CRR0)

                # Liquefaction severity index (LSI)
                FoS = CRR/CSR
                
                if FoS <= 1.411:
                    P_L = 1 / (1 + ((FoS / 0.96) ** 4.5))
                else:
                    P_L = 0
                
                if row[1] <= 20:
                    w = 10-(0.5*row[1])
                else:
                    w = 0
                
                LSI = P_L * w * (row[1] -depth)                
                
            depth = row[1]
            gamma = row[6]

            # Set output
            output.append([row[1], ce, cb, cr, cs, N60, sigmav, sigmavp, CN, N160, delta_n, N160cs, rd, CSR, MSF, k_sigma, CRR0, CRR, FoS, P_L, w, LSI])
                                    
        self.new_bore_log_data = pd.DataFrame(output, columns=['depth', 'ce', 'cb', 'cr', 'cs', 'N60', 'sigmav', 'sigmavp', 'CN', 'N160', 'delta_n', 'N160cs', 'rd', 'CSR', 'MSF', 'k_simga', 'CRR0', 'CRR', 'FS', 'P_L', 'w', 'LSI'])
               
        # LSI classification (Sonmez and Gokceoglu, 2005)
        sumLSI = self.new_bore_log_data.LSI.sum()
        
        if (sumLSI == 0):
            self.categoryLSI = "No liquefaction"
        elif (sumLSI > 0 and sumLSI < 15):
            self.categoryLSI = "Very low"
        elif (sumLSI >= 15 and sumLSI < 35):
            self.categoryLSI = "Low"
        elif (sumLSI >= 35 and sumLSI < 65):
            self.categoryLSI = "Moderate"
        elif (sumLSI >= 65 and sumLSI < 85):
            self.categoryLSI = "High"
        elif (sumLSI >= 85 and sumLSI < 100):
            self.categoryLSI = "Very high"
    
    # visualization of the liquefaction analysis
    def visualize(self):
        fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
        [ax[x].xaxis.tick_top() for x in range(ax.shape[0])]
        [ax[x].xaxis.set_label_position('top') for x in range(ax.shape[0])]

        total_depth = max(self.new_bore_log_data['depth'])*-1.1

        # subplot of CSR & CRR
        depth_0 = 0
        layer_change_0 = 0
        liquefiable_0 = False
        csr_0 = 0
        crr_0 = 0
        na_0 = False

        csrcrr_plot_max_x = 1
        fos_plot_max_x = self.new_bore_log_data.loc[self.new_bore_log_data.loc[:, 'FS'] != 'n.a.', 'FS'].max()*1.1
        
        for i, row in self.new_bore_log_data.iterrows():
            depth_1 = -row[0]
            ax[0].plot([0, csrcrr_plot_max_x], [depth_1, depth_1], **self.viz_dashed_guidelines)
            ax[1].plot([0, fos_plot_max_x], [depth_1, depth_1], **self.viz_dashed_guidelines)
            na_1 = False
            csr_1 = row['CSR']
            crr_1 = row['CRR']
            
            if row['FS'] == 'n.a.':
                na_1 = True
                liquefiable_1 = False
            elif row['FS'] > self.fs_threshold: 
                liquefiable_1 = False
            else:
                liquefiable_1 = True
                    
            if i > 0:
                if not na_1 and not na_0:
                    ax[0].plot([csr_0, csr_1], [depth_0, depth_1], 'k--')
                    ax[0].plot([crr_0, crr_1], [depth_0, depth_1], 'k-')

            if not liquefiable_0 == liquefiable_1:
                layer_change_1 = (depth_1+depth_0)*.5
                if not liquefiable_1:
                    ax[1].text(0.5*fos_plot_max_x, (layer_change_0+layer_change_1)*0.5, 'LIQUEFIED ZONE', **Borehole.viz_liquefied_text_kwargs)
                else:
                    ax[1].text(0.5*fos_plot_max_x, (layer_change_0 + layer_change_1) * 0.5, 'NON-LIQUEFIED ZONE', **Borehole.viz_liquefied_text_kwargs)

                ax[0].plot([-1, csrcrr_plot_max_x], [(depth_1+depth_0)*.5, (depth_1+depth_0)*.5], color=(0, 0, 0, 0.15))
                ax[1].plot([0, fos_plot_max_x], [(depth_1 + depth_0) * .5, (depth_1 + depth_0) * .5], color=(0, 0, 0, 0.15))
                layer_change_0 = layer_change_1

            liquefiable_0 = liquefiable_1
            depth_0 = depth_1
            csr_0 = csr_1
            crr_0 = crr_1
            na_0 = na_1

        if liquefiable_1:
            ax[1].text(0.5*fos_plot_max_x, (total_depth+layer_change_1)*0.5, 'LIQUEFIED ZONE', **Borehole.viz_liquefied_text_kwargs)
        else:
            ax[1].text(0.5 * fos_plot_max_x, (total_depth + layer_change_1) * 0.5, 'NON-LIQUEFIED ZONE', **Borehole.viz_liquefied_text_kwargs)

        ax[0].plot([0, 0], [0, 0], 'k--', label='CSR')
        ax[0].plot([0, 0], [0, 0], 'k-', label='Earthquake-induced CRR')
        ax[0].legend(loc='lower right')
        ax[0].set(xlabel='CSR & CRR', xlim=[0, csrcrr_plot_max_x])
        ax[0].set_ylim(top=0, bottom=-20)

        # subplot of Factor of safety
        depth_0 = 0
        fs_0 = 0
        for i, row in self.new_bore_log_data.iterrows():
            fs_1 = row['FS']
            depth_1 = -row['depth']
            if i > 0 and fs_1 != 'n.a.' and fs_0 != 'n.a.':
                ax[1].plot([fs_0, fs_1], [depth_0, depth_1], 'k-')

            fs_0 = fs_1
            depth_0 = depth_1
            
        ax[1].plot([self.fs_threshold, self.fs_threshold], [0, total_depth], '--', color=(0, 0, 0, 0.1))
        ax[1].set(xlabel='FACTOR OF SAFETY', xlim=[0, fos_plot_max_x])
        ax[1].set_ylim(top=0, bottom=-20)
            
        if self.name != None:
            fig.suptitle(self.name, fontsize=14, y=.99)
        plt.show()


    def save_to_file(self, file_name):
        self.new_bore_log_data.to_excel(file_name + '.xlsx')
        print(file_name + '.xls has been saved.')
