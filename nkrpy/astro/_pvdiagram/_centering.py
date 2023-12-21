    if test:
        print('Imageshape (post mask; should be smaller): ', image.shape)
        for i in range(int(image.shape[0] / 4), int(image.shape[0] / 1.5), 3):
            testax.imshow(image[i, :, :], origin='lower', cmap='cividis', interpolation='nearest')
            testax.axhline(int(image.shape[1] / 2), color='white', linestyle='--')
            testax.set_aspect(1./testax.get_data_ratio())
            testfig.savefig(f'testpv/test_cut-{i}.pdf', bbox_inches='tight')
            testax.cla()
    timer.log()
    embed() if config['interactive'] else None
    # fix rotation
    offsets = [0] if not config['fixpa'] else range(-5, 5, 1)
    if config['fixcen'] or config['fixpa'] or test:
        new_freqcoord = vel_array[~vel_array_mask] * 1000. * 100. / constants.c* rf + rf
        rotf, rota = plt.subplots()
    for offset in offsets:
        offset /= 2.
        rotated_image = rotate_image(np.nan_to_num(image), axis=-1, deg=pa + offset, use_skimage=True)
        if test or config['fixpa']:
            print(f'Trying PA: {pa + offset}')
            title = '' if offset != 0 else '(default)'
            rot_mm, _ = momentmap(rotated_image.T, freq_axis=new_freqcoord, moment=0)
            rota.cla()
            rota.set_title(f'Moment 0 Map, PA: {pa + offset} {title}')
            rota.imshow(rot_mm, origin='lower', cmap='cividis', interpolation='nearest')
            rota.axhline(int(rot_mm.shape[1] / 2), color='white', linestyle='--')
            rota.axvline(int(rot_mm.shape[0] / 2), color='white', linestyle='--')
            rota.set_aspect(1./rota.get_data_ratio())
        if config['fixpa']:
            rotf.show()
            answer = input('Press [ret] to continue or [space] to accept this pa')
            if answer.replace(' ', '') == '' and answer != "": # if accept
                pa += offset
                config['pa'] = pa
                break
        elif test:
            rotf.savefig(f'testpv/rest_rot-{pa + offset}.pdf', bbox_inches='tight')
    else:
        print('No replacement found, using config value.')
    if test:
        print('Rotated image: ', rotated_image.shape)
    # fix dec center
    offsets = [0] if not config['fixcen'] else range(0, 20, 1)
    for offset in offsets:
        offset -= 10
        offset /= 10.
        pixoffset = int(offset / 3600. / wcs.get('ra---sin')['del'])
        if test or config['fixcen']:
            print(f'Trying center dec shift: {offset}" or {pixoffset} pixels')
            shift_rotated_image = shift(rotated_image, shift=[0, 0, pixoffset])
            rot_mm, _ = momentmap(shift_rotated_image.T, freq_axis=new_freqcoord, moment=0)
            rota.cla()
            title = '' if pixoffset != 0 else '(default)'
            rota.set_title(f'Moment 0 Map, DEC Offset {title}: {offset}" or {pixoffset} pixels')
            rota.imshow(rot_mm, origin='lower', cmap='cividis', interpolation='nearest')
            rota.axhline(int(rot_mm.shape[1] / 2), color='white', linestyle='--')
            rota.axvline(int(rot_mm.shape[0] / 2), color='white', linestyle='--')
            rota.set_aspect(1./rota.get_data_ratio())
        if config['fixcen']:
            rotf.show()
            answer = input('Press [ret] to continue or [space] to accept this center')
            if answer.replace(' ', '') == '' and answer != "": # if accept
                wcs.shift_axis(axis='ra---sin', unit='pix', val=pixoffset)
                rotated_image = shift_rotated_image
                config['dec'] = config['dec'] + offset / 3600.
                break
        elif test:
            rotf.savefig(f'testpv/rest_dec_center-{pa + offset}.pdf', bbox_inches='tight')
    else:
        print('No replacement found, using config value.')
    # fix ra center
    offsets = [0] if not config['fixcen'] else range(0, 20, 1)
    for offset in offsets:
        offset -= 10
        offset /= 10.
        pixoffset = int(offset / 3600. / wcs.get('ra---sin')['del'])
        if test or config['fixcen']:
            print(f'Trying center ra shift: {offset}" or {pixoffset} pixels')
            shift_rotated_image = shift(rotated_image, shift=[0, pixoffset, 0])
            rot_mm, _ = momentmap(shift_rotated_image.T, freq_axis=new_freqcoord, moment=0)
            rota.cla()
            title = '' if offset != 0 else '(default)'
            rota.set_title(f'Moment 0 Map, RA Offset: {offset}" or {pixoffset} pixels  {title}')
            rota.imshow(rot_mm, origin='lower', cmap='cividis', interpolation='nearest')
            rota.axhline(int(rot_mm.shape[1] / 2), color='white', linestyle='--')
            rota.axvline(int(rot_mm.shape[0] / 2), color='white', linestyle='--')
            rota.set_aspect(1./rota.get_data_ratio())
        if config['fixcen']:
            rotf.show()
            answer = input('Press [ret] to continue or [space] to accept this center')
            if answer.replace(' ', '') == '' and answer != "": # if accept
                wcs.shift_axis(axis='dec--sin', unit='pix', val=pixoffset)
                rotated_image = shift_rotated_image
                config['ra'] = config['ra'] + offset / 3600.
                rotf.clf()
                break
        elif test:
            rotf.savefig(f'testpv/rest_dec_center-{pa + offset}.pdf', bbox_inches='tight')
    else:
        rotf.clf()
        print('No replacement found, using config value.')
    if config['fixcen'] or config['fixpa']:
        rotf.clf()