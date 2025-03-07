
! ***********************************************************************
!
!   Copyright (C) 2017-2019  Rob Farmer & The MESA Team
!
!   this file is part of mesa.
!
!   mesa is free software; you can redistribute it and/or modify
!   it under the terms of the gnu general library public license as published
!   by the free software foundation; either version 2 of the license, or
!   (at your option) any later version.
!
!   mesa is distributed in the hope that it will be useful,
!   but without any warranty; without even the implied warranty of
!   merchantability or fitness for a particular purpose.  see the
!   gnu library general public license for more details.
!
!   you should have received a copy of the gnu library general public license
!   along with this software; if not, write to the free software
!   foundation, inc., 59 temple place, suite 330, boston, ma 02111-1307 usa
!
! ***********************************************************************





module run_star_extras

  use star_lib
  use star_def
  use const_def
  use math_lib
  use auto_diff
  use colors_lib

  implicit none


!DEFINE ALL GLOCBAL VARIABLE HERE



  include "test_suite_extras_def.inc"

  ! these routines are called by the standard run_star check_model
  contains

  include "test_suite_extras.inc"


  subroutine extras_controls(id, ierr)
    integer, intent(in) :: id
    integer, intent(out) :: ierr
    type (star_info), pointer :: s
    ierr = 0
    call star_ptr(id, s, ierr)
    if (ierr /= 0) return
       print *, "Extras startup routine"

    call process_color_files(id, ierr)
    s% extras_startup => extras_startup
    s% extras_check_model => extras_check_model
    s% extras_finish_step => extras_finish_step
    s% extras_after_evolve => extras_after_evolve
    s% how_many_extra_history_columns => how_many_extra_history_columns
    s% data_for_extra_history_columns => data_for_extra_history_columns
    s% how_many_extra_profile_columns => how_many_extra_profile_columns
    s% data_for_extra_profile_columns => data_for_extra_profile_columns

    print *, "Sellar atmosphere:", s% x_character_ctrl(1)
    print *, "Instrument:", s% x_character_ctrl(2)

  end subroutine extras_controls





!###########################################################
!## THINGS I HAVE NOT TOUCHED
!###########################################################

  subroutine process_color_files(id, ierr)
    integer, intent(in) :: id
    integer, intent(out) :: ierr
    type(star_info), pointer :: s
    integer :: i

    ierr = 0
    call star_ptr(id, s, ierr)
    if (ierr /= 0) return

  end subroutine process_color_files


  subroutine extras_startup(id, restart, ierr)
     integer, intent(in) :: id
     logical, intent(in) :: restart
     integer, intent(out) :: ierr
     type (star_info), pointer :: s
     ierr = 0
     call star_ptr(id, s, ierr)
     if (ierr /= 0) return
     call test_suite_startup(s, restart, ierr)
  end subroutine extras_startup


  subroutine extras_after_evolve(id, ierr)
     integer, intent(in) :: id
     integer, intent(out) :: ierr
     type (star_info), pointer :: s
     real(dp) :: dt
     ierr = 0
     call star_ptr(id, s, ierr)
     if (ierr /= 0) return

     write(*,'(a)') 'finished custom colors'

     call test_suite_after_evolve(s, ierr)

  end subroutine extras_after_evolve


  ! returns either keep_going, retry, or terminate.
  integer function extras_check_model(id)
     integer, intent(in) :: id
     integer :: ierr
     type (star_info), pointer :: s
     ierr = 0
     call star_ptr(id, s, ierr)
     if (ierr /= 0) return
     extras_check_model = keep_going
  end function extras_check_model


  INTEGER FUNCTION how_many_extra_profile_columns(id)
     USE star_def, ONLY: star_info
     INTEGER, INTENT(IN) :: id

     INTEGER :: ierr
     TYPE(star_info), POINTER :: s

     ierr = 0
     CALL star_ptr(id, s, ierr)
     IF (ierr /= 0) RETURN

     how_many_extra_profile_columns = 0
  END FUNCTION how_many_extra_profile_columns


  SUBROUTINE data_for_extra_profile_columns(id, n, nz, names, vals, ierr)
     USE star_def, ONLY: star_info, maxlen_profile_column_name
     USE const_def, ONLY: DP
     INTEGER, INTENT(IN) :: id, n, nz
     CHARACTER(LEN=maxlen_profile_column_name) :: names(n)
     REAL(DP) :: vals(nz, n)
     INTEGER, INTENT(OUT) :: ierr

     TYPE(star_info), POINTER :: s

     ierr = 0
     CALL star_ptr(id, s, ierr)
     IF (ierr /= 0) RETURN

  END SUBROUTINE data_for_extra_profile_columns


  ! Returns either keep_going, retry, or terminate
  INTEGER FUNCTION extras_finish_step(id)
     USE chem_def
     INTEGER, INTENT(IN) :: id

     INTEGER :: ierr
     TYPE(star_info), POINTER :: s

     ierr = 0
     CALL star_ptr(id, s, ierr)
     IF (ierr /= 0) RETURN

     extras_finish_step = keep_going
  END FUNCTION extras_finish_step






!###########################################################
!## MESA STUFF
!###########################################################

  !FUNCTIONS FOR OPENING LOOKUP FILE AND FINDING THE NUMBER OF FILES AND THIER FILE PATHS
  integer function how_many_extra_history_columns(id)
      ! Determines how many extra history columns are added based on a file
      integer, intent(in) :: id
      integer :: ierr, n
      character(len=100), allocatable :: strings(:)
      type(star_info), pointer :: s

      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) then
          how_many_extra_history_columns = 0
          return
      end if

      ! Read strings from the file
      call read_strings_from_file(strings, n, id)

      ! Number of columns is the size of the strings array
      how_many_extra_history_columns = n + 2

      !print *, "This many columns added to history file:", n

      if (allocated(strings)) deallocate(strings)
  end function how_many_extra_history_columns


  function basename(path) result(base)
      ! Extracts the base name from a given file path
      character(len=*), intent(in) :: path
      character(len=512) :: base
      integer :: last_slash

      ! Find the position of the last slash
      last_slash = len_trim(path)
      do while (last_slash > 0 .and. path(last_slash:last_slash) /= '/')
          last_slash = last_slash - 1
      end do

      ! Extract the base name
      base = path(last_slash+1:)
  end function basename

  function remove_dat(path) result(base)
      ! Extracts the portion of the string after the first dot
      character(len=*), intent(in) :: path
      character(len=512) :: base
      integer :: first_dot

      ! Find the position of the first dot
      first_dot = 0
      do while (first_dot < len_trim(path) .and. path(first_dot+1:first_dot+1) /= '.')
          first_dot = first_dot + 1
      end do

      ! Check if an dot was found
      if (first_dot < len_trim(path)) then
          ! Extract the part after the dot
          base = path(:first_dot)
      else
          ! No dot found, return the input string
          base = path
      end if
  end function remove_dat


  subroutine read_strings_from_file(strings, n, id)
      ! Reads strings from a file into an allocatable array
      integer, intent(in) :: id
      character(len=512) :: filename
      character(len=100), allocatable :: strings(:)
      integer, intent(out) :: n
      integer :: unit, i, status
      character(len=100) :: line
      integer :: ierr
      type(star_info), pointer :: s

      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return

      ! Construct the filename
      filename = trim(s%x_character_ctrl(2)) // "/" // trim(basename(s%x_character_ctrl(2)))

      ! Initialize
      n = 0

      ! Open the file
      unit = 10
      open(unit, file=filename, status='old', action='read', iostat=status)
      if (status /= 0) then
          print *, "Error: Could not open file", filename
          stop
      end if

      ! Count lines in the file to determine the size of the array
      do
          read(unit, '(A)', iostat=status) line
          if (status /= 0) exit
          n = n + 1  ! for bolometric correctionms
      end do
      rewind(unit)

      ! Allocate the array and read the strings
      if (allocated(strings)) deallocate(strings)
      allocate(strings(n))
      do i = 1, n
          read(unit, '(A)') strings(i)
      end do

      close(unit)
  end subroutine read_strings_from_file



  subroutine data_for_extra_history_columns(id, n, names, vals, ierr)
      ! Populates data for the extra history columns
      integer, intent(in) :: id, n
      integer, intent(out) :: ierr
      character(len=maxlen_history_column_name) :: names(n)
      real(dp) :: vals(n)
      type(star_info), pointer :: s
      integer :: i, num_strings
      character(len=100), allocatable :: array_of_strings(:)
      real(dp) :: teff, log_g, metallicity, R, d,  bolometric_magnitude, bolometric_flux
      character(len=256) :: sed_filepath, filter_filepath, filter_name, filter_dir, vega_filepath
      real(dp), dimension(:), allocatable :: wavelengths, fluxes, filter_wavelengths, filter_trans
      logical :: make_sed

      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return

      ! Extract input parameters
      !    mesa/star_data/public/star_data_step_work.inc
      teff = s%T(1)
      log_g = LOG10(s%grav(1))
      R = s%R(1)  ! * 1d3
      metallicity = s%Z(1)! mass fraction metals
      !print * , metallicity
      !stop
      d = s%job%extras_rpar(1)

      sed_filepath = s%x_character_ctrl(1)
      filter_dir = s%x_character_ctrl(2)
      vega_filepath = s%x_character_ctrl(3)
      make_sed = trim(adjustl(s%x_character_ctrl(4))) == 'true'

      ! Read filters from file
      if (allocated(array_of_strings)) deallocate(array_of_strings)
      allocate(array_of_strings(n))
      call read_strings_from_file(array_of_strings, num_strings, id)

      !PRINT *, "################################################"

      ! Compute bolometric values
      CALL CalculateBolometric(teff, log_g, metallicity, R, d,  bolometric_magnitude, bolometric_flux, wavelengths, fluxes, sed_filepath)
      names(1) = "Mag_bol"
      vals(1) = bolometric_magnitude
      names(2) = "Flux_bol"
      vals(2) = bolometric_flux

      ! Populate history columns
      if (allocated(array_of_strings)) then
          do i = 3, how_many_extra_history_columns(id)
              filter_name = "Unknown"
              if (i <= num_strings + 2) filter_name = trim(remove_dat(array_of_strings(i - 2)))
              names(i) = filter_name
              filter_filepath = trim(filter_dir) // "/" // array_of_strings(i - 2)

              if (teff >= 0 .and. log_g >= 0 .and. metallicity >= 0) then
                  vals(i) = CalculateSynthetic(teff, log_g, metallicity, ierr, wavelengths, fluxes, filter_wavelengths, filter_trans, filter_filepath, vega_filepath, array_of_strings(i - 2), make_sed)
                  if (ierr /= 0) vals(i) = -1.0_dp
              else
                  vals(i) = -1.0_dp
                  ierr = 1
              end if
              !PRINT *, names(i), vals(i)
          end do
      else
          ierr = 1  ! Indicate an error if array_of_strings is not allocated
      end if

      if (allocated(array_of_strings)) deallocate(array_of_strings)
  end subroutine data_for_extra_history_columns




!###########################################################
!## CUSTOM COLOURS
!###########################################################

!****************************
!Calculate Bolometric Photometry Using Multiple SEDs
!****************************

  SUBROUTINE CalculateBolometric(teff, log_g, metallicity, R, d, bolometric_magnitude, bolometric_flux, wavelengths, fluxes, sed_filepath)
    REAL(8), INTENT(IN) :: teff, log_g, metallicity, R, d
    CHARACTER(LEN=*), INTENT(IN) :: sed_filepath
    REAL(DP), INTENT(OUT) :: bolometric_magnitude, bolometric_flux

    REAL (8), ALLOCATABLE :: lu_logg(:), lu_meta(:), lu_teff(:)
    CHARACTER(LEN=100), ALLOCATABLE :: file_names(:)
    REAL, DIMENSION(:,:), ALLOCATABLE :: lookup_table
    REAL(DP), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: wavelengths, fluxes
    CHARACTER(LEN=256) :: lookup_file

    lookup_file = TRIM(sed_filepath) // '/lookup_table.csv'

    ! Call to load the lookup table
    CALL LoadLookupTable(lookup_file, lookup_table, file_names, lu_logg, lu_meta, lu_teff)
    !print *, 'logg', lu_logg
    !print *,  'meta', lu_meta
    !print *, 'teff', lu_teff
    ! Interpolate Spectral Energy Distribution
    !CALL ConstructSED_Robust(teff, log_g, metallicity, R, d, file_names, lu_teff, lu_logg, lu_meta, sed_filepath, wavelengths, fluxes)
    CALL ConstructSED(teff, log_g, metallicity, R, d, file_names, lu_teff, lu_logg, lu_meta, sed_filepath, wavelengths, fluxes)

    ! Calculate bolometric flux and magnitude
    CALL CalculateBolometricPhot(wavelengths, fluxes, bolometric_magnitude, bolometric_flux)
  END SUBROUTINE CalculateBolometric



!****************************
!Construct SED With Combination of SEDs
!****************************

SUBROUTINE ConstructSED(teff, log_g, metallicity, R, d, file_names, lu_teff, lu_logg, lu_meta, stellar_model_dir, wavelengths, fluxes)
  REAL(8), INTENT(IN) :: teff, log_g, metallicity, R, d
  REAL(8), INTENT(IN) :: lu_teff(:), lu_logg(:), lu_meta(:)
  CHARACTER(LEN=*), INTENT(IN) :: stellar_model_dir
  CHARACTER(LEN=100), INTENT(IN) :: file_names(:)
  REAL(DP), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: wavelengths, fluxes

  INTEGER, DIMENSION(4) :: closest_indices
  REAL(DP), DIMENSION(:), ALLOCATABLE :: temp_wavelengths, temp_flux, common_wavelengths
  REAL(DP), DIMENSION(:,:), ALLOCATABLE :: model_fluxes
  REAL(DP), DIMENSION(4) :: weights, distances
  INTEGER :: i, n_points
  REAL(DP) :: sum_weights
  REAL(DP), DIMENSION(:), ALLOCATABLE :: diluted_flux

  ! Get the four closest stellar models
  CALL GetClosestStellarModels(teff, log_g, metallicity, lu_teff, lu_logg, lu_meta, closest_indices)

  ! Load the first SED to define the wavelength grid
  CALL LoadSED(TRIM(stellar_model_dir) // TRIM(file_names(closest_indices(1))), closest_indices(1), temp_wavelengths, temp_flux)
  n_points = SIZE(temp_wavelengths)
  ALLOCATE(common_wavelengths(n_points))
  common_wavelengths = temp_wavelengths

  ! Allocate flux array for the models (4 models, n_points each)
  ALLOCATE(model_fluxes(4, n_points))
  CALL InterpolateArray(temp_wavelengths, temp_flux, common_wavelengths, model_fluxes(1, :))

  ! Load and interpolate remaining SEDs
  DO i = 2, 4
    CALL LoadSED(TRIM(stellar_model_dir) // TRIM(file_names(closest_indices(i))), closest_indices(i), temp_wavelengths, temp_flux)
    CALL InterpolateArray(temp_wavelengths, temp_flux, common_wavelengths, model_fluxes(i, :))
  END DO

  ! Compute distances and weights for the four models
  DO i = 1, 4
    distances(i) = SQRT((lu_teff(closest_indices(i)) - teff)**2 + &
                        (lu_logg(closest_indices(i)) - log_g)**2 + &
                        (lu_meta(closest_indices(i)) - metallicity)**2)
    IF (distances(i) == 0.0) distances(i) = 1.0E-10  ! Prevent division by zero
    weights(i) = 1.0 / distances(i)
  END DO

  ! Normalize weights
  sum_weights = SUM(weights)
  weights = weights / sum_weights

  ! Allocate output arrays
  ALLOCATE(wavelengths(n_points), fluxes(n_points))
  wavelengths = common_wavelengths
  fluxes = 0.0

  ! Perform weighted combination of the model fluxes (still at the stellar surface)
  DO i = 1, 4
    fluxes = fluxes + weights(i) * model_fluxes(i, :)
  END DO

  ! Now, apply the dilution factor (R/d)^2 to convert the surface flux density
  ! into the observed flux density at Earth.
  ALLOCATE(diluted_flux(n_points))
  CALL dilute_flux(fluxes, R, d, diluted_flux)
  fluxes = diluted_flux

  ! Deallocate temporary arrays
  DEALLOCATE(temp_wavelengths, temp_flux, common_wavelengths, diluted_flux)

END SUBROUTINE ConstructSED





SUBROUTINE dilute_flux(surface_flux, R, d, calibrated_flux)
  ! Define the double precision kind if not already defined
  INTEGER, PARAMETER :: DP = KIND(1.0D0)

  ! Input: surface_flux is an array of flux values at the stellar surface
  REAL(DP), INTENT(IN)  :: surface_flux(:)
  REAL(DP), INTENT(IN)  :: R, d  ! R = stellar radius, d = distance (both in the same units, e.g., cm)

  ! Output: calibrated_flux will be the flux observed at Earth
  REAL(DP), INTENT(OUT) :: calibrated_flux(:)

  ! Check that the output array has the same size as the input
  IF (SIZE(calibrated_flux) /= SIZE(surface_flux)) THEN
    PRINT *, "Error in dilute_flux: Output array must have the same size as input array."
    STOP 1
  END IF

  ! Apply the dilution factor (R/d)^2 to each element
  calibrated_flux = surface_flux * ( (R / d)**2 )

END SUBROUTINE dilute_flux






!****************************
!Identify The Four Closest Stellar Models
!****************************

SUBROUTINE GetClosestStellarModels(teff, log_g, metallicity, lu_teff, lu_logg, lu_meta, closest_indices)
  REAL(8), INTENT(IN) :: teff, log_g, metallicity
  REAL(8), INTENT(IN) :: lu_teff(:), lu_logg(:), lu_meta(:)
  INTEGER, DIMENSION(4), INTENT(OUT) :: closest_indices

  INTEGER :: i, n, j
  REAL(DP) :: distance, norm_teff, norm_logg, norm_meta
  REAL(DP), DIMENSION(:), ALLOCATABLE :: scaled_lu_teff, scaled_lu_logg, scaled_lu_meta
  REAL(DP), DIMENSION(4) :: min_distances
  INTEGER, DIMENSION(4) :: indices
  REAL(DP) :: teff_min, teff_max, logg_min, logg_max, meta_min, meta_max, teff_dist, logg_dist, meta_dist

  n = SIZE(lu_teff)
  min_distances = HUGE(1.0)
  indices = -1

  ! Find min and max for normalization
  teff_min = MINVAL(lu_teff)
  teff_max = MAXVAL(lu_teff)
  logg_min = MINVAL(lu_logg)
  logg_max = MAXVAL(lu_logg)
  meta_min = MINVAL(lu_meta)
  meta_max = MAXVAL(lu_meta)

  ! Allocate and scale lookup table values
  ALLOCATE(scaled_lu_teff(n), scaled_lu_logg(n), scaled_lu_meta(n))

  IF (teff_max - teff_min > 0.00) THEN
    scaled_lu_teff = (lu_teff - teff_min) / (teff_max - teff_min)
  END IF

  IF (logg_max - logg_min > 0.00) THEN
    scaled_lu_logg = (lu_logg - logg_min) / (logg_max - logg_min)
  END IF

  IF (meta_max - meta_min > 0.00) THEN
    scaled_lu_meta = (lu_meta - meta_min) / (meta_max - meta_min)
  END IF

  ! Normalize input parameters
  norm_teff = (teff - teff_min) / (teff_max - teff_min)
  norm_logg = (log_g - logg_min) / (logg_max - logg_min)
  norm_meta = (metallicity - meta_min) / (meta_max - meta_min)

  ! Debug: !PRINT normalized input parameters
  !PRINT *, "Normalized parameters for target:"
  !PRINT *, "  teff = ", teff, "  logg = ", log_g, "  meta = ", metallicity, n

  ! Find closest models
  DO i = 1, n

    teff_dist = 0.0
    logg_dist = 0.0
    meta_dist = 0.0

    IF (teff_max - teff_min > 0.00) THEN
      teff_dist = scaled_lu_teff(i) - norm_teff
    END IF

    IF (logg_max - logg_min > 0.00) THEN
      logg_dist = scaled_lu_logg(i) - norm_logg
    END IF

    IF (meta_max - meta_min > 0.00) THEN
      meta_dist = scaled_lu_meta(i) - norm_meta
    END IF


    !distance = SQRT(teff_dist**2 + logg_dist**2 + meta_dist**2)
    distance = teff_dist**2 + logg_dist**2 + meta_dist**2   !SQRT is a monotonic transform so pointless?

    DO j = 1, 4
      IF (distance < min_distances(j)) THEN
        ! Shift larger distances down
        IF (j < 4) THEN
          min_distances(j+1:4) = min_distances(j:3)
          indices(j+1:4) = indices(j:3)
        END IF
        min_distances(j) = distance
        indices(j) = i
        EXIT
      END IF
    END DO
  END DO

  closest_indices = indices
  ! Deallocate arrays
  DEALLOCATE(scaled_lu_teff, scaled_lu_logg, scaled_lu_meta)
END SUBROUTINE GetClosestStellarModels




!****************************
!Calculate Bolometric Magnitude and Flux
!****************************

  SUBROUTINE CalculateBolometricPhot(wavelengths, fluxes, bolometric_magnitude, bolometric_flux)
    REAL(DP), DIMENSION(:), INTENT(INOUT) :: wavelengths, fluxes
    REAL(DP), INTENT(OUT) :: bolometric_magnitude, bolometric_flux
    INTEGER :: i

    ! Validate inputs and replace invalid wavelengths with 0
    DO i = 1, SIZE(wavelengths) - 1
      IF (wavelengths(i) <= 0.0 .OR. fluxes(i) < 0.0) THEN
        PRINT *, "bolometric Invalid input at index", i, ":", wavelengths(i), fluxes(i)
        fluxes(i) = 0.0  ! Replace invalid wavelength with 0
      END IF
    END DO


    ! Perform trapezoidal integration
    ! Debug: Print the first few wavelengths and flux values
    !PRINT *, "Wavelengths (first 5):", wavelengths(1:MIN(5, SIZE(wavelengths)))
    !PRINT *, "Fluxes (first 5):", fluxes(1:MIN(5, SIZE(fluxes)))

    ! Call trapezoidal integration
    CALL RombergIntegration(wavelengths, fluxes, bolometric_flux)

    ! Debug: Check the integration result
    !PRINT *, "Integrated Flux:", bolometric_flux

    ! Validate integration result
    IF (bolometric_flux <= 0.0) THEN
      PRINT *, "Error: Flux integration resulted in non-positive value."
      bolometric_magnitude = 99.0
      RETURN
    END IF

        ! Calculate bolometric magnitude
    IF (bolometric_flux <= 0.0) THEN
      PRINT *, "Error: Flux integration resulted in non-positive value."
      bolometric_magnitude = 99.0
      RETURN
    ELSE IF (bolometric_flux < 1.0E-10) THEN
      PRINT *, "Warning: Flux value is very small, precision might be affected."
    END IF

  bolometric_magnitude = FluxToMagnitude(bolometric_flux)

  END SUBROUTINE CalculateBolometricPhot












!###########################################################
!## Synthetic Photometry
!###########################################################

!****************************
!Calculate Synthetic Photometry Using SED and Filter
!****************************



REAL(DP) FUNCTION CalculateSynthetic(temperature, gravity, metallicity, ierr, wavelengths, fluxes, filter_wavelengths, filter_trans, filter_filepath, vega_filepath, filter_name, make_sed)

    ! Input arguments
    REAL(DP), INTENT(IN) :: temperature, gravity, metallicity
    CHARACTER(LEN=*), INTENT(IN) :: filter_filepath, filter_name, vega_filepath
    INTEGER, INTENT(OUT) :: ierr
    CHARACTER(LEN=1000) :: line

    REAL(DP), DIMENSION(:), INTENT(INOUT) :: wavelengths, fluxes
    REAL(DP), DIMENSION(:), ALLOCATABLE, INTENT(INOUT) :: filter_wavelengths, filter_trans
    LOGICAL, INTENT(IN) :: make_sed
    LOGICAL :: dir_exists
    ! Local variables
    REAL(DP), DIMENSION(:), ALLOCATABLE :: convolved_flux, interpolated_filter
    CHARACTER(LEN=100) :: csv_file
    REAL(DP) :: synthetic_magnitude, synthetic_flux, vega_flux
    INTEGER :: max_size, i
    REAL(DP) :: magnitude
    REAL(DP) :: wv, fl, cf, fwv, ftr

    csv_file =  'LOGS/SED/' //TRIM(remove_dat(filter_name)) // '_SED.csv'
    ! Initialize error flag
    ierr = 0

    ! Load filter data
    CALL LoadFilter(filter_filepath, filter_wavelengths, filter_trans)

    ! Check for invalid gravity input
    IF (gravity <= 0.0_DP) THEN
        ierr = 1
        CalculateSynthetic = -1.0_DP
        RETURN
    END IF

    ! Allocate interpolated_filter if not already allocated
    IF (.NOT. ALLOCATED(interpolated_filter)) THEN
        ALLOCATE(interpolated_filter(SIZE(wavelengths)))
        interpolated_filter = 0.0_DP
    END IF

      ! Perform SED convolution
  ALLOCATE(convolved_flux(SIZE(wavelengths)))
  CALL ConvolveSED(wavelengths, fluxes, filter_wavelengths, filter_trans, convolved_flux)

  IF (make_sed) THEN
! Determine the maximum size among all arrays
max_size = MAX(SIZE(wavelengths), SIZE(filter_wavelengths), SIZE(fluxes), SIZE(convolved_flux), SIZE(filter_trans))

! Open the CSV file for writing
OPEN(UNIT=10, FILE=csv_file, STATUS='REPLACE', ACTION='WRITE', IOSTAT=ierr)
IF (ierr /= 0) THEN
    PRINT *, "Error opening file for writing"
    STOP
END IF

! Write headers to the CSV file
WRITE(10, '(A)') "wavelengths,fluxes,convolved_flux,filter_wavelengths,filter_trans"

! Loop through data and safely write values, ensuring no out-of-bounds errors
DO i = 1, max_size
    ! Initialize values to zero in case they are out of bounds
    wv = 0.0_DP
    fl = 0.0_DP
    cf = 0.0_DP
    fwv = 0.0_DP
    ftr = 0.0_DP

    ! Assign actual values only if within valid indices
    IF (i <= SIZE(wavelengths)) wv = wavelengths(i)
    IF (i <= SIZE(fluxes)) fl = fluxes(i)
    IF (i <= SIZE(convolved_flux)) cf = convolved_flux(i)
    IF (i <= SIZE(filter_wavelengths)) fwv = filter_wavelengths(i)
    IF (i <= SIZE(filter_trans)) ftr = filter_trans(i)

    ! Write the formatted output
    WRITE(line, '(ES14.6, ",", ES14.6, ",", ES14.6, ",", ES14.6, ",", ES14.6)') &
        wv, fl, cf, fwv, ftr
    WRITE(10, '(A)') TRIM(line)
END DO

! Close the file
CLOSE(10)

  END IF

    ! Inform the user of successful writing
    !PRINT *, "Data written to ", csv_file
    vega_flux = CalculateVegaFlux(vega_filepath, filter_wavelengths, filter_trans, filter_name, make_sed)

    ! Calculate synthetic flux and magnitude
    CALL CalculateSyntheticFlux(wavelengths, convolved_flux, synthetic_flux, filter_wavelengths, filter_trans)

    !PRINT *, "VEGA zero point:", vega_flux

    IF (vega_flux > 0.0_DP) THEN
      CalculateSynthetic = -2.5 * LOG10(synthetic_flux / vega_flux)
    ELSE
      PRINT *, "Error: Vega flux is zero, magnitude calculation is invalid."
      CalculateSynthetic = HUGE(1.0_DP)
    END IF

END FUNCTION CalculateSynthetic




!****************************
!Convolve SED With Filter
!****************************

  SUBROUTINE ConvolveSED(wavelengths, fluxes, filter_wavelengths, filter_trans, convolved_flux)
    REAL(DP), DIMENSION(:), INTENT(INOUT) :: wavelengths, fluxes
    REAL(DP), DIMENSION(:), INTENT(INOUT) :: filter_wavelengths, filter_trans
    REAL(DP), DIMENSION(:), ALLOCATABLE :: convolved_flux
    REAL(DP), DIMENSION(:), ALLOCATABLE :: interpolated_filter
    INTEGER :: n

    n = SIZE(wavelengths)

    ! Allocate arrays
    ALLOCATE(interpolated_filter(n))
    !ALLOCATE(convolved_flux(n))

    ! Interpolate the filter transmission onto the wavelengths array
    CALL InterpolateArray(filter_wavelengths, filter_trans, wavelengths, interpolated_filter)

    ! Perform convolution (element-wise multiplication)
    convolved_flux = fluxes * interpolated_filter

    ! Deallocate arrays (optional, depending on context)
    DEALLOCATE(interpolated_filter)
  END SUBROUTINE ConvolveSED



!****************************
!Calculate Synthetic Flux and Magnitude
!****************************
  SUBROUTINE CalculateSyntheticFlux(wavelengths, fluxes, synthetic_flux, filter_wavelengths, filter_trans)
    REAL(DP), DIMENSION(:), INTENT(IN) :: wavelengths, fluxes
    REAL(DP), DIMENSION(:), INTENT(INOUT) :: filter_wavelengths, filter_trans
    REAL(DP), INTENT(OUT) :: synthetic_flux
    INTEGER :: i
    REAL(DP) :: integrated_flux, integrated_filter
    CHARACTER(LEN=256) :: vega_filepath



    ! Validate inputs
    DO i = 1, SIZE(wavelengths) - 1
      IF (wavelengths(i) <= 0.0 .OR. fluxes(i) < 0.0) THEN
        PRINT *, "synthetic Invalid input at index", i, ":", wavelengths(i), fluxes(i)
        STOP
      END IF
    END DO

    CALL RombergIntegration(wavelengths, fluxes* wavelengths, integrated_flux)
    CALL RombergIntegration(filter_wavelengths, filter_trans * filter_wavelengths, integrated_filter)

    ! Store the total flux
    IF (integrated_filter > 0.0) THEN
        synthetic_flux = integrated_flux / integrated_filter
    ELSE
        PRINT *, "Error: Integrated filter transmission is zero."
        synthetic_flux = -1.0_DP
        RETURN
    END IF

  END SUBROUTINE CalculateSyntheticFlux



  REAL(DP) FUNCTION FluxToMagnitude(flux)
    REAL(DP), INTENT(IN) :: flux
    !print *, 'flux:', flux
    IF (flux <= 0.0) THEN
      PRINT *, "Error: Flux must be positive to calculate magnitude."
      FluxToMagnitude = 99.0  ! Return an error value
    ELSE
      FluxToMagnitude = -2.5 * LOG10(flux)
    END IF
  END FUNCTION FluxToMagnitude






FUNCTION CalculateVegaFlux(vega_filepath, filt_wave, filt_trans, filter_name, make_sed) RESULT(vega_flux)
  CHARACTER(LEN=*), INTENT(IN) :: vega_filepath, filter_name
  CHARACTER(len = 100) :: output_csv
  REAL(DP), DIMENSION(:), INTENT(INOUT) :: filt_wave, filt_trans
  REAL(DP) :: vega_flux
  REAL(DP) :: int_flux, int_filter
  REAL(DP), ALLOCATABLE :: vega_wave(:), vega_flux_arr(:), conv_flux(:)
  LOGICAL, INTENT(IN) :: make_sed
  INTEGER :: i, unit, max_size
  REAL(DP) :: wv, fl, cf, fwv, ftr
  INTEGER:: ierr
  CHARACTER(LEN=1000) :: line

  ! Load the Vega SED using the custom routine.
  CALL LoadVegaSED(vega_filepath, vega_wave, vega_flux_arr)

  ! Convolve the Vega SED with the filter transmission.
  CALL ConvolveSED(vega_wave, vega_flux_arr, filt_wave, filt_trans, conv_flux)

  ! Integrate the convolved Vega SED and the filter transmission.
  CALL RombergIntegration(vega_wave, vega_wave*conv_flux, int_flux)
  CALL RombergIntegration(filt_wave, filt_wave*filt_trans, int_filter)

  IF (int_filter > 0.0_DP) THEN
    vega_flux = int_flux / int_filter
  ELSE
    vega_flux = -1.0_DP
  END IF




  IF (make_sed) THEN
    ! Determine the maximum size among all arrays
    max_size = MAX(SIZE(vega_wave), SIZE(vega_flux_arr), SIZE(conv_flux), SIZE(filt_wave), SIZE(filt_trans))


    output_csv = 'LOGS/SED/VEGA_' //TRIM(remove_dat(filter_name)) // '_SED.csv'

    ! Open the CSV file for writing
    OPEN(UNIT=10, FILE=output_csv, STATUS='REPLACE', ACTION='WRITE', IOSTAT=ierr)
    IF (ierr /= 0) THEN
        PRINT *, "Error opening file for writing"
        STOP
    END IF

    WRITE(10, '(A)') "wavelengths,fluxes,convolved_flux,filter_wavelengths,filter_trans"


    ! Loop through data and safely write values, ensuring no out-of-bounds errors
    DO i = 1, max_size
        ! Initialize values to zero in case they are out of bounds
        wv = 0.0_DP
        fl = 0.0_DP
        cf = 0.0_DP
        fwv = 0.0_DP
        ftr = 0.0_DP

        ! Assign actual values only if within valid indices
        IF (i <= SIZE(vega_wave)) wv = vega_wave(i)
        IF (i <= SIZE(vega_flux_arr)) fl = vega_flux_arr(i)
        IF (i <= SIZE(conv_flux)) cf = conv_flux(i)
        IF (i <= SIZE(filt_wave)) fwv = filt_wave(i)
        IF (i <= SIZE(filt_trans)) ftr = filt_trans(i)

        ! Write the formatted output
        WRITE(line, '(ES14.6, ",", ES14.6, ",", ES14.6, ",", ES14.6, ",", ES14.6)') &
            wv, fl, cf, fwv, ftr
        WRITE(10, '(A)') TRIM(line)


    END DO

    ! Close the file
    CLOSE(10)

  END IF





  DEALLOCATE(conv_flux, vega_wave, vega_flux_arr)
END FUNCTION CalculateVegaFlux










SUBROUTINE LoadVegaSED(filepath, wavelengths, flux)
  CHARACTER(LEN=*), INTENT(IN) :: filepath
  REAL(DP), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: wavelengths, flux
  CHARACTER(LEN=512) :: line
  INTEGER :: unit, n_rows, status, i
  REAL(DP) :: temp_wave, temp_flux

  unit = 20
  OPEN(unit, FILE=TRIM(filepath), STATUS='OLD', ACTION='READ', IOSTAT=status)
  IF (status /= 0) THEN
    PRINT *, "Error: Could not open Vega SED file ", TRIM(filepath)
    STOP
  END IF

  ! Skip header line.
  READ(unit, '(A)', IOSTAT=status) line
  IF (status /= 0) THEN
    PRINT *, "Error: Could not read header from Vega SED file ", TRIM(filepath)
    STOP
  END IF

  ! Count the number of data lines.
  n_rows = 0
  DO
    READ(unit, '(A)', IOSTAT=status) line
    IF (status /= 0) EXIT
    n_rows = n_rows + 1
  END DO

  REWIND(unit)
  READ(unit, '(A)', IOSTAT=status) line  ! Skip header again

  ALLOCATE(wavelengths(n_rows))
  ALLOCATE(flux(n_rows))

  i = 0
  DO
    READ(unit, *, IOSTAT=status) temp_wave, temp_flux  ! Ignore any extra columns.
    IF (status /= 0) EXIT
    i = i + 1
    wavelengths(i) = temp_wave
    flux(i) = temp_flux
  END DO

  CLOSE(unit)
END SUBROUTINE LoadVegaSED




!###########################################################
!## FILE IO
!###########################################################


!****************************
!Load Filter File
!****************************

  SUBROUTINE LoadFilter(directory, filter_wavelengths, filter_trans)
    CHARACTER(LEN=*), INTENT(IN) :: directory
    REAL(DP), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: filter_wavelengths, filter_trans

    CHARACTER(LEN=512) :: line
    INTEGER :: unit, n_rows, status, i
    REAL :: temp_wavelength, temp_trans

    ! Open the file
    unit = 20
    OPEN(unit, FILE=TRIM(directory), STATUS='OLD', ACTION='READ', IOSTAT=status)
    IF (status /= 0) THEN
      PRINT *, "Error: Could not open file ", TRIM(directory)
      STOP
    END IF

    ! Skip header line
    READ(unit, '(A)', IOSTAT=status) line
    IF (status /= 0) THEN
      PRINT *, "Error: Could not read the file", TRIM(directory)
      STOP
    END IF

    ! Count rows in the file
    n_rows = 0
    DO
      READ(unit, '(A)', IOSTAT=status) line
      IF (status /= 0) EXIT
      n_rows = n_rows + 1
    END DO

    ! Allocate arrays
    ALLOCATE(filter_wavelengths(n_rows))
    ALLOCATE(filter_trans(n_rows))

    ! Rewind to the first non-comment line
    REWIND(unit)
    DO
      READ(unit, '(A)', IOSTAT=status) line
      IF (status /= 0) THEN
        PRINT *, "Error: Could not rewind file", TRIM(directory)
        STOP
      END IF
      IF (line(1:1) /= "#") EXIT
    END DO

    ! Read and parse data
    i = 0
    DO
      READ(unit, *, IOSTAT=status) temp_wavelength, temp_trans
      IF (status /= 0) EXIT
      i = i + 1

      filter_wavelengths(i) = temp_wavelength
      filter_trans(i) = temp_trans
    END DO

    CLOSE(unit)
  END SUBROUTINE LoadFilter


!****************************
!Load Lookup Table For Identifying Stellar Atmosphere Models
!****************************


  SUBROUTINE LoadLookupTable(lookup_file, lookup_table, out_file_names, out_logg, out_meta, out_teff)
    CHARACTER(LEN=*), INTENT(IN) :: lookup_file
    REAL, DIMENSION(:,:), ALLOCATABLE, INTENT(OUT) :: lookup_table
    CHARACTER(LEN=100), ALLOCATABLE, INTENT(INOUT) :: out_file_names(:)
    REAL(8), ALLOCATABLE, INTENT(INOUT) :: out_logg(:), out_meta(:), out_teff(:)

    INTEGER :: i, n_rows, status, unit
    CHARACTER(LEN=512) :: line
    CHARACTER(LEN=*), PARAMETER :: delimiter = ","
    CHARACTER(LEN=100), ALLOCATABLE :: columns(:), headers(:)
    INTEGER :: logg_col, meta_col, teff_col

    ! Open the file
    unit = 10
    OPEN(unit, FILE=lookup_file, STATUS='old', ACTION='read', IOSTAT=status)
    IF (status /= 0) THEN
      PRINT *, "Error: Could not open file", lookup_file
      STOP
    END IF

    ! Read header line
    READ(unit, '(A)', IOSTAT=status) line
    IF (status /= 0) THEN
      PRINT *, "Error: Could not read header line"
      STOP
    END IF

    CALL SplitLine(line, delimiter, headers)

    ! Determine column indices for logg, meta, and teff
    logg_col = GetColumnIndex(headers, "logg")
    teff_col = GetColumnIndex(headers, "teff")

    meta_col = GetColumnIndex(headers, "meta")
    IF (meta_col < 0) THEN
      meta_col = GetColumnIndex(headers, "feh")
    END IF

    n_rows = 0
    DO
      READ(unit, '(A)', IOSTAT=status) line
      IF (status /= 0) EXIT
      n_rows = n_rows + 1
    END DO
    REWIND(unit)

    ! Skip header
    READ(unit, '(A)', IOSTAT=status) line

    ! Allocate output arrays
    ALLOCATE(out_file_names(n_rows))
    ALLOCATE(out_logg(n_rows), out_meta(n_rows), out_teff(n_rows))

    ! Read and parse the file
    i = 0
    DO
      READ(unit, '(A)', IOSTAT=status) line
      IF (status /= 0) EXIT
      i = i + 1

      CALL SplitLine(line, delimiter, columns)

      ! Populate arrays
      out_file_names(i) = columns(1)
      !PRINT *, columns

      IF (logg_col > 0) THEN
        IF (columns(logg_col) /= "") THEN
          READ(columns(logg_col), *) out_logg(i)
        ELSE
          out_logg(i) = 0.0
        END IF
      ELSE
        out_logg(i) = 0.0
      END IF

      IF (meta_col > 0) THEN
        IF (columns(meta_col) /= "") THEN
          READ(columns(meta_col), *) out_meta(i)
        ELSE
          out_meta(i) = 0.0
        END IF
      ELSE
        out_meta(i) = 0.0
      END IF

      IF (teff_col > 0) THEN
        IF (columns(teff_col) /= "") THEN
          READ(columns(teff_col), *) out_teff(i)
        ELSE
          out_teff(i) = 0.0
        END IF
      ELSE
        out_teff(i) = 0.0
      END IF

    END DO

    CLOSE(unit)

  CONTAINS

    FUNCTION GetColumnIndex(headers, target) RESULT(index)
      CHARACTER(LEN=100), INTENT(IN) :: headers(:)
      CHARACTER(LEN=*), INTENT(IN) :: target
      INTEGER :: index, i
      CHARACTER(LEN=100) :: clean_header, clean_target

      index = -1
      clean_target = TRIM(ADJUSTL(target))  ! Clean the target string

      DO i = 1, SIZE(headers)
        clean_header = TRIM(ADJUSTL(headers(i)))  ! Clean each header
        IF (clean_header == clean_target) THEN
          index = i
          EXIT
        END IF
      END DO
    END FUNCTION GetColumnIndex

    SUBROUTINE SplitLine(line, delimiter, tokens)
      CHARACTER(LEN=*), INTENT(IN) :: line, delimiter
      CHARACTER(LEN=100), ALLOCATABLE, INTENT(OUT) :: tokens(:)
      INTEGER :: num_tokens, pos, start, len_delim

      len_delim = LEN_TRIM(delimiter)
      start = 1
      num_tokens = 0
      IF (ALLOCATED(tokens)) DEALLOCATE(tokens)

      DO
        pos = INDEX(line(start:), delimiter)

        IF (pos == 0) EXIT
        num_tokens = num_tokens + 1
        CALL AppendToken(tokens, line(start:start + pos - 2))
        start = start + pos + len_delim - 1
      END DO

      num_tokens = num_tokens + 1
      CALL AppendToken(tokens, line(start:))
    END SUBROUTINE SplitLine

    SUBROUTINE AppendToken(tokens, token)
      CHARACTER(LEN=*), INTENT(IN) :: token
      CHARACTER(LEN=100), ALLOCATABLE, INTENT(INOUT) :: tokens(:)
      CHARACTER(LEN=100), ALLOCATABLE :: temp(:)
      INTEGER :: n

      IF (.NOT. ALLOCATED(tokens)) THEN
        ALLOCATE(tokens(1))
        tokens(1) = token
      ELSE
        n = SIZE(tokens)
        ALLOCATE(temp(n))
        temp = tokens  ! Backup the current tokens
        DEALLOCATE(tokens)  ! Deallocate the old array
        ALLOCATE(tokens(n + 1))  ! Allocate with one extra space
        tokens(1:n) = temp  ! Restore old tokens
        tokens(n + 1) = token  ! Add the new token
        DEALLOCATE(temp)  ! Clean up temporary array
      END IF
    END SUBROUTINE AppendToken

  END SUBROUTINE LoadLookupTable






  !###########################################################
  !## MATHS
  !###########################################################

!****************************
!Trapezoidal and Simpson Integration For Flux Calculation
!****************************

  SUBROUTINE TrapezoidalIntegration(x, y, result)
    REAL(DP), DIMENSION(:), INTENT(IN) :: x, y
    REAL(DP), INTENT(OUT) :: result

    INTEGER :: i, n
    REAL :: sum

    n = SIZE(x)
    sum = 0.0

    ! Validate input sizes
    IF (SIZE(x) /= SIZE(y)) THEN
      PRINT *, "Error: x and y arrays must have the same size."
      STOP
    END IF

    IF (SIZE(x) < 2) THEN
      PRINT *, "Error: x and y arrays must have at least 2 elements."
      STOP
    END IF

    ! Perform trapezoidal integration
    DO i = 1, n - 1
      sum = sum + 0.5 * (x(i + 1) - x(i)) * (y(i + 1) + y(i))
    END DO

    result = sum
  END SUBROUTINE TrapezoidalIntegration


SUBROUTINE SimpsonIntegration(x, y, result)
  INTEGER, PARAMETER :: DP = KIND(1.0D0)
  REAL(DP), DIMENSION(:), INTENT(IN) :: x, y
  REAL(DP), INTENT(OUT) :: result

  INTEGER :: i, n
  REAL(DP) :: sum, h1, h2, f1, f2, f0

  n = SIZE(x)
  sum = 0.0_DP

  ! Validate input sizes
  IF (SIZE(x) /= SIZE(y)) THEN
    PRINT *, "Error: x and y arrays must have the same size."
    STOP
  END IF

  IF (SIZE(x) < 2) THEN
    PRINT *, "Error: x and y arrays must have at least 2 elements."
    STOP
  END IF

  ! Perform adaptive Simpson’s rule
  DO i = 1, n - 2, 2
    h1 = x(i+1) - x(i)       ! Step size for first interval
    h2 = x(i+2) - x(i+1)     ! Step size for second interval

    f0 = y(i)
    f1 = y(i+1)
    f2 = y(i+2)

    ! Simpson's rule: (h/3) * (f0 + 4f1 + f2)
    sum = sum + (h1 + h2) / 6.0_DP * (f0 + 4.0_DP * f1 + f2)
  END DO

  ! Handle the case where n is odd (last interval)
  IF (MOD(n,2) == 0) THEN
    sum = sum + 0.5_DP * (x(n) - x(n-1)) * (y(n) + y(n-1))
  END IF

  result = sum
END SUBROUTINE SimpsonIntegration

SUBROUTINE RombergIntegration(x, y, result)
  INTEGER, PARAMETER :: DP = KIND(1.0D0)
  REAL(DP), DIMENSION(:), INTENT(IN) :: x, y
  REAL(DP), INTENT(OUT) :: result

  INTEGER :: i, j, k, n, m
  REAL(DP), DIMENSION(:), ALLOCATABLE :: R
  REAL(DP) :: h, sum, factor

  n = SIZE(x)
  m = INT(LOG(REAL(n, DP)) / LOG(2.0_DP)) + 1  ! Number of refinement levels

  ! Validate input sizes
  IF (SIZE(x) /= SIZE(y)) THEN
    PRINT *, "Error: x and y arrays must have the same size."
    STOP
  END IF

  IF (n < 2) THEN
    PRINT *, "Error: x and y arrays must have at least 2 elements."
    STOP
  END IF

  ALLOCATE(R(m))

  ! Compute initial trapezoidal rule estimate
  h = x(n) - x(1)
  R(1) = 0.5_DP * h * (y(1) + y(n))

  ! Refinement using Romberg's method
  DO j = 2, m
    sum = 0.0_DP
    DO i = 1, 2**(j-2)
      sum = sum + y(1 + (2*i - 1) * (n-1) / (2**(j-1)))
    END DO

    h = h / 2.0_DP
    R(j) = 0.5_DP * R(j-1) + h * sum

    ! Richardson extrapolation
    factor = 4.0_DP
    DO k = j, 2, -1
      R(k-1) = (factor * R(k) - R(k-1)) / (factor - 1.0_DP)
      factor = factor * 4.0_DP
    END DO
  END DO

  result = R(1)
  DEALLOCATE(R)
END SUBROUTINE RombergIntegration

!****************************
!Linear Interpolation For SED Construction
!****************************

!will be removed in the future so long as binary search proves consistently faster
  SUBROUTINE LinearInterpolate_linearsearch(x, y, x_val, y_val)
    REAL(DP), INTENT(IN) :: x(:), y(:), x_val
    REAL(DP), INTENT(OUT) :: y_val
    INTEGER :: i
    REAL(DP) :: slope

    ! Validate input sizes
    IF (SIZE(x) < 2) THEN
      PRINT *, "Error: x array has fewer than 2 points."
      y_val = 0.0_DP
      RETURN
    END IF

    IF (SIZE(x) /= SIZE(y)) THEN
      PRINT *, "Error: x and y arrays have different sizes."
      y_val = 0.0_DP
      RETURN
    END IF

    ! Handle out-of-bounds cases
    IF (x_val < MINVAL(x)) THEN
      y_val = y(1)
      RETURN
    ELSE IF (x_val > MAXVAL(x)) THEN
      y_val = y(SIZE(y))
      RETURN
    END IF

    ! Perform interpolation
    DO i = 1, SIZE(x) - 1
      IF (x_val >= x(i) .AND. x_val <= x(i + 1)) THEN
        slope = (y(i + 1) - y(i)) / (x(i + 1) - x(i))
        y_val = y(i) + slope * (x_val - x(i))
        RETURN
      END IF
    END DO

    y_val = 0.0_DP
  END SUBROUTINE LinearInterpolate_linearsearch


SUBROUTINE LinearInterpolate(x, y, x_val, y_val)
  REAL(DP), INTENT(IN) :: x(:), y(:), x_val
  REAL(DP), INTENT(OUT) :: y_val
  INTEGER :: low, high, mid

  ! Validate input sizes
  IF (SIZE(x) < 2) THEN
    PRINT *, "Error: x array has fewer than 2 points."
    y_val = 0.0_DP
    RETURN
  END IF

  IF (SIZE(x) /= SIZE(y)) THEN
    PRINT *, "Error: x and y arrays have different sizes."
    y_val = 0.0_DP
    RETURN
  END IF

  ! Handle out-of-bounds cases
  IF (x_val <= x(1)) THEN
    y_val = y(1)
    RETURN
  ELSE IF (x_val >= x(SIZE(x))) THEN
    y_val = y(SIZE(y))
    RETURN
  END IF

  ! Binary search to find the proper interval [x(low), x(low+1)]
  low = 1
  high = SIZE(x)
  DO WHILE (high - low > 1)
    mid = (low + high) / 2
    IF (x(mid) <= x_val) THEN
      low = mid
    ELSE
      high = mid
    END IF
  END DO

  ! Linear interpolation between x(low) and x(low+1)
  y_val = y(low) + (y(low+1) - y(low)) / (x(low+1) - x(low)) * (x_val - x(low))
END SUBROUTINE LinearInterpolate




!****************************
!Array Interpolation For SED Construction
!****************************

  SUBROUTINE InterpolateArray(x_in, y_in, x_out, y_out)
    REAL(DP), INTENT(IN) :: x_in(:), y_in(:), x_out(:)
    REAL(DP), INTENT(OUT) :: y_out(:)
    INTEGER :: i

    ! Validate input sizes
    IF (SIZE(x_in) < 2 .OR. SIZE(y_in) < 2) THEN
      PRINT *, "Error: x_in or y_in arrays have fewer than 2 points."
      STOP
    END IF

    IF (SIZE(x_in) /= SIZE(y_in)) THEN
      PRINT *, "Error: x_in and y_in arrays have different sizes."
      STOP
    END IF

    IF (SIZE(x_out) <= 0) THEN
      PRINT *, "Error: x_out array is empty."
      STOP
    END IF

    DO i = 1, SIZE(x_out)
      CALL LinearInterpolate(x_in, y_in, x_out(i), y_out(i))
    END DO
  END SUBROUTINE InterpolateArray





FUNCTION det3(M) RESULT(d)
  IMPLICIT NONE
  REAL(8), INTENT(IN) :: M(3,3)
  REAL(8) :: d
  d = M(1,1)*(M(2,2)*M(3,3) - M(2,3)*M(3,2)) - &
      M(1,2)*(M(2,1)*M(3,3) - M(2,3)*M(3,1)) + &
      M(1,3)*(M(2,1)*M(3,2) - M(2,2)*M(3,1))
END FUNCTION det3

SUBROUTINE ComputeBarycentrics(P, P0, P1, P2, P3, bary)
  IMPLICIT NONE
  REAL(8), INTENT(IN) :: P(3), P0(3), P1(3), P2(3), P3(3)
  REAL(8), INTENT(OUT) :: bary(4)
  REAL(8) :: M(3,3), d, d0, d1, d2, d3
  REAL(8) :: rhs(3)

  ! Build matrix M with columns = P1-P0, P2-P0, P3-P0
  M(:,1) = P1 - P0
  M(:,2) = P2 - P0
  M(:,3) = P3 - P0

  d = det3(M)
  IF (ABS(d) < 1.0D-12) THEN
    bary = -1.0D0  ! signal degenerate
    RETURN
  END IF

  ! Solve M * [u, v, w]^T = P - P0 using Cramer's rule
  rhs = P - P0
  d0 = det3(reshape([rhs(1), M(1,2), M(1,3), &
                      rhs(2), M(2,2), M(2,3), &
                      rhs(3), M(3,2), M(3,3)], [3,3]))
  d1 = det3(reshape([M(1,1), rhs(1), M(1,3), &
                      M(2,1), rhs(2), M(2,3), &
                      M(3,1), rhs(3), M(3,3)], [3,3]))
  d2 = det3(reshape([M(1,1), M(1,2), rhs(1), &
                      M(2,1), M(2,2), rhs(2), &
                      M(3,1), M(3,2), rhs(3)], [3,3]))
  ! The barycentrics: w0 = 1 - u - v - w, w1 = u, w2 = v, w3 = w.
  bary(2) = d0/d
  bary(3) = d1/d
  bary(4) = d2/d
  bary(1) = 1.0D0 - bary(2) - bary(3) - bary(4)
END SUBROUTINE ComputeBarycentrics


SUBROUTINE FindEnclosingSimplex(teff, log_g, metallicity, lu_teff, lu_logg, lu_meta, &
                                  simplex_indices, bary_weights)
  IMPLICIT NONE
  INTEGER, PARAMETER :: DP = KIND(1.0D0)
  REAL(8), INTENT(IN) :: teff, log_g, metallicity
  REAL(8), INTENT(IN) :: lu_teff(:), lu_logg(:), lu_meta(:)
  INTEGER, ALLOCATABLE, INTENT(OUT) :: simplex_indices(:)
  REAL(DP), ALLOCATABLE, INTENT(OUT) :: bary_weights(:)
  INTEGER :: i, num_points, j, temp_index, k
  REAL(8), ALLOCATABLE :: dists(:)
  REAL(8), DIMENSION(3) :: P, P0, P1, P2, P3
  REAL(8), DIMENSION(4) :: bary
  REAL(8) :: tol
  REAL(8) :: sumw, temp_w(4)

  tol = 1.0D3
  num_points = SIZE(lu_teff)
  ALLOCATE(dists(num_points))
  
  ! Compute distances for each point
  DO i = 1, num_points
    dists(i) = SQRT((lu_teff(i)-teff)**2 + (lu_logg(i)-log_g)**2 + (lu_meta(i)-metallicity)**2)
  END DO
  
  !print *, dists
  !stop

  ! Find indices of the 4 smallest distances (simple selection sort for 4 elements)
  ALLOCATE(simplex_indices(4))
  DO i = 1, 4
    simplex_indices(i) = i
  END DO
  DO i = 5, num_points
    IF (dists(i) < dists(simplex_indices(4))) THEN
      simplex_indices(4) = i
      ! Re-sort the 4 indices by distance (simple bubble sort)
      DO j = 1, 3
         IF (dists(simplex_indices(j)) > dists(simplex_indices(j+1))) THEN
            temp_index = simplex_indices(j)
            simplex_indices(j) = simplex_indices(j+1)
            simplex_indices(j+1) = temp_index
         END IF
      END DO
    END IF
  END DO

  ! Now we have 4 candidate vertices. Form their coordinates.
  P(1) = teff;    P(2) = log_g;    P(3) = metallicity
  P0 = [ lu_teff(simplex_indices(1)), lu_logg(simplex_indices(1)), lu_meta(simplex_indices(1)) ]
  P1 = [ lu_teff(simplex_indices(2)), lu_logg(simplex_indices(2)), lu_meta(simplex_indices(2)) ]
  P2 = [ lu_teff(simplex_indices(3)), lu_logg(simplex_indices(3)), lu_meta(simplex_indices(3)) ]
  P3 = [ lu_teff(simplex_indices(4)), lu_logg(simplex_indices(4)), lu_meta(simplex_indices(4)) ]
  
CALL ComputeBarycentrics(P, P0, P1, P2, P3, bary)
IF ( ANY(bary < -tol) ) THEN
  PRINT *, "Warning: Degenerate tetrahedron. Using inverse-distance weighting fallback."
  ALLOCATE(bary_weights(4))
  sumw = 0.0D0
  DO k = 1, 4
    temp_w(k) = 1.0D0 / (dists(simplex_indices(k)) + 1.0D-12)
    sumw = sumw + temp_w(k)
  END DO
  bary_weights = temp_w / sumw
  RETURN
ELSE
  ALLOCATE(bary_weights(4))
  bary_weights = bary
  RETURN
END IF
END SUBROUTINE FindEnclosingSimplex





!--------------------------------------------------------------------
! A robust SED interpolation using scattered data interpolation.
! Ideally, this uses Delaunay triangulation with barycentric interpolation.
! The subroutine FindEnclosingSimplex is the heart of the method.
!--------------------------------------------------------------------
SUBROUTINE ConstructSED_Robust(teff, log_g, metallicity, R, d,  &
         file_names, lu_teff, lu_logg, lu_meta, stellar_model_dir,  &
         wavelengths, fluxes)
  IMPLICIT NONE
  INTEGER, PARAMETER :: DP = KIND(1.0D0)
  ! Inputs
  REAL(8), INTENT(IN) :: teff, log_g, metallicity, R, d
  REAL(8), INTENT(IN) :: lu_teff(:), lu_logg(:), lu_meta(:)
  CHARACTER(LEN=*), INTENT(IN) :: stellar_model_dir
  CHARACTER(LEN=100), INTENT(IN) :: file_names(:)
  ! Outputs
  REAL(DP), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: wavelengths, fluxes

  ! Local variables
  INTEGER, ALLOCATABLE, DIMENSION(:) :: simplex_indices
  REAL(DP), ALLOCATABLE, DIMENSION(:) :: bary_weights
  INTEGER :: num_vertices, n_points, i
  REAL(DP), DIMENSION(:), ALLOCATABLE :: common_wavelengths
  REAL(DP), DIMENSION(:), ALLOCATABLE :: interp_flux, temp_wavelength, temp_flux

  !--------------------------------------------------------------------
  ! Step 1: Find the simplex that encloses (teff, log_g, metallicity)
  CALL FindEnclosingSimplex(teff, log_g, metallicity, lu_teff, lu_logg, lu_meta, &
                            simplex_indices, bary_weights)
  num_vertices = SIZE(simplex_indices)
  PRINT *, "Enclosing simplex indices: ", simplex_indices
  PRINT *, "Barycentric weights: ", bary_weights

  !--------------------------------------------------------------------
  ! Step 2: Define a common wavelength grid.
  ! If we have only one vertex (nearest neighbor fallback), just use that SED.
  CALL LoadSED(TRIM(stellar_model_dir)//TRIM(file_names(simplex_indices(1))), &
               simplex_indices(1), temp_wavelength, temp_flux)
  n_points = SIZE(temp_wavelength)
  IF (n_points <= 0) THEN
    PRINT *, "Error: Loaded SED from ", TRIM(file_names(simplex_indices(1))), " has no wavelengths."
    STOP
  END IF
  ALLOCATE(common_wavelengths(n_points))
  common_wavelengths = temp_wavelength
  DEALLOCATE(temp_wavelength, temp_flux)

  !--------------------------------------------------------------------
  ! Step 3: Compute the SED.
  ALLOCATE(interp_flux(n_points))
  interp_flux = 0.0D0
  IF (num_vertices == 1) THEN
    ! Nearest neighbor: just load the SED without interpolation.
    CALL LoadAndInterpolateSED(TRIM(stellar_model_dir)//TRIM(file_names(simplex_indices(1))), simplex_indices(1), common_wavelengths, interp_flux)
  ELSE
    DO i = 1, num_vertices
      !print *, TRIM(stellar_model_dir)//TRIM(file_names(simplex_indices(i))), simplex_indices(i), common_wavelengths, temp_flux
      CALL LoadAndInterpolateSED(TRIM(stellar_model_dir)//TRIM(file_names(simplex_indices(i))), simplex_indices(i), common_wavelengths, temp_flux)
      ! Check that temp_flux has the expected size:
      IF (SIZE(temp_flux) /= n_points) THEN
         PRINT *, "Error: SED from ", TRIM(file_names(simplex_indices(i))), " has mismatched wavelength grid."
         STOP
      END IF
      interp_flux = interp_flux + bary_weights(i) * temp_flux
      DEALLOCATE(temp_flux)
    END DO
  END IF

  !--------------------------------------------------------------------
  ! Step 4: Apply the dilution factor (R/d)^2.
  ALLOCATE(fluxes(n_points))
  CALL dilute_flux(interp_flux, R, d, fluxes)
  ALLOCATE(wavelengths(n_points))
  wavelengths = common_wavelengths

  ! Clean up
  DEALLOCATE(common_wavelengths, interp_flux)
  
END SUBROUTINE ConstructSED_Robust




SUBROUTINE LoadAndInterpolateSED(filename, index, common_wavelengths, flux_out)
  INTEGER, PARAMETER :: DP = KIND(1.0D0)
  CHARACTER(LEN=*), INTENT(IN) :: filename
  INTEGER, INTENT(IN) :: index
  REAL(DP), INTENT(IN) :: common_wavelengths(:)
  REAL(DP), ALLOCATABLE, INTENT(OUT) :: flux_out(:)

  REAL(DP), DIMENSION(:), ALLOCATABLE :: temp_wavelengths, temp_flux
  INTEGER :: n, i, n_print

  ! Load the SED from the file.
  CALL LoadSED(TRIM(filename), index, temp_wavelengths, temp_flux)
  
  ! Check that the loaded data arrays have at least 2 points.
  IF (SIZE(temp_wavelengths) < 2 .OR. SIZE(temp_flux) < 2) THEN
    PRINT *, "Error: Loaded SED arrays are too small."
    STOP
  END IF

  ! Allocate flux_out to match the size of the common wavelength grid.
  n = SIZE(common_wavelengths)
  ALLOCATE(flux_out(n))
  
  ! Interpolate the loaded SED onto the common wavelength grid.
  CALL InterpolateArray(temp_wavelengths, temp_flux, common_wavelengths, flux_out)
  
  ! Print the first five common wavelengths.
  n_print = MIN(5, SIZE(common_wavelengths))
  PRINT *, "First ", n_print, " common wavelengths:"
  DO i = 1, n_print
    PRINT *, common_wavelengths(i)
  END DO
  
  ! Print the first five interpolated flux values.
  n_print = MIN(5, SIZE(flux_out))
  PRINT *, "First ", n_print, " interpolated flux values:"
  DO i = 1, n_print
    PRINT *, flux_out(i)
  END DO

  ! Clean up temporary arrays.
  DEALLOCATE(temp_wavelengths, temp_flux)
END SUBROUTINE LoadAndInterpolateSED

!****************************
!Load SED File
!****************************

  SUBROUTINE LoadSED(directory, index, wavelengths, flux)
    CHARACTER(LEN=*), INTENT(IN) :: directory
    INTEGER, INTENT(IN) :: index
    REAL(DP), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: wavelengths, flux

    CHARACTER(LEN=512) :: line
    INTEGER :: unit, n_rows, status, i
    REAL(DP) :: temp_wavelength, temp_flux

    ! Open the file
    unit = 20
    OPEN(unit, FILE=TRIM(directory), STATUS='OLD', ACTION='READ', IOSTAT=status)
    IF (status /= 0) THEN
      PRINT *, "Error: Could not open file ", TRIM(directory)
      STOP
    END IF

    ! Skip header lines
    DO
      READ(unit, '(A)', IOSTAT=status) line
      IF (status /= 0) THEN
        PRINT *, "Error: Could not read the file", TRIM(directory)
        STOP
      END IF
      IF (line(1:1) /= "#") EXIT
    END DO

    ! Count rows in the file
    n_rows = 0
    DO
      READ(unit, '(A)', IOSTAT=status) line
      IF (status /= 0) EXIT
      n_rows = n_rows + 1
    END DO

    ! Allocate arrays
    ALLOCATE(wavelengths(n_rows))
    ALLOCATE(flux(n_rows))

    ! Rewind to the first non-comment line
    REWIND(unit)
    DO
      READ(unit, '(A)', IOSTAT=status) line
      IF (status /= 0) THEN
        PRINT *, "Error: Could not rewind file", TRIM(directory)
        STOP
      END IF
      IF (line(1:1) /= "#") EXIT
    END DO

    ! Read and parse data
    i = 0
    DO
      READ(unit, *, IOSTAT=status) temp_wavelength, temp_flux
      IF (status /= 0) EXIT
      i = i + 1
      ! Convert f_lambda to f_nu
      wavelengths(i) = temp_wavelength
      flux(i) = temp_flux
    END DO

    CLOSE(unit)

  END SUBROUTINE LoadSED





end module run_star_extras
