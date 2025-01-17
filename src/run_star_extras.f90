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
     IMPLICIT NONE
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
     IMPLICIT NONE
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
     IMPLICIT NONE
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
      implicit none
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
          n = n + 1 ! for bolometric correctionms
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


  !FUNCTIONS FOR POPULATING MESA HISTORY FILE
  subroutine data_for_extra_history_columns(id, n, names, vals, ierr)
      ! Populates data for the extra history columns
      integer, intent(in) :: id, n
      character(len=maxlen_history_column_name) :: names(n)
      real(dp) :: vals(n)
      integer, intent(out) :: ierr
      type(star_info), pointer :: s
      integer :: i
      character(len=100), allocatable :: array_of_strings(:)
      integer :: num_strings
      REAL(dp) :: teff, log_g, metallicity
      CHARACTER(LEN=256) :: sed_filepath, filter_filepath, filter_name, filter_dir
      REAL :: bolometric_magnitude, bolometric_flux
      REAL(DP), DIMENSION(:), ALLOCATABLE :: wavelengths, fluxes, filter_wavelengths, filter_trans
      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return

      ! Input parameters
      teff = s%T(1)
      log_g = LOG10(s%grav(1))
      metallicity = s%job%extras_rpar(1)
      sed_filepath = s%x_character_ctrl(1)
      filter_dir = s%x_character_ctrl(2)      


      ! Populate array_of_strings
      if (allocated(array_of_strings)) deallocate(array_of_strings)
      allocate(array_of_strings(n))
      call read_strings_from_file(array_of_strings, num_strings, id)
      print *, array_of_strings, num_strings, id
      !STOP

      CALL CalculateBolometricMagnitude(teff, log_g, metallicity, bolometric_magnitude, bolometric_flux,wavelengths, fluxes, sed_filepath)
      names(1) = "Mag_bol"
      vals(1) = bolometric_magnitude 

      names(2) = "Flux_bol"
      vals(2) = bolometric_flux 
      !print *, '1111111111111111', wavelengths, SIZE(wavelengths)
      !print *, teff, log_g, metallicity
      !STOP
      ! Validate and populate values
      if (allocated(array_of_strings)) then
          do i = 3, how_many_extra_history_columns(id)
              if (i <= num_strings+2) then
                  filter_name = trim(remove_dat(array_of_strings(i-2)))
              else
                  filter_name = "Unknown"
              end if

              names(i) = filter_name
              !print *, '2222222222222222', wavelengths, SIZE(wavelengths)
              ! Prepend filter name with filter dir to generate filepath
              filter_filepath = trim(filter_dir) // "/" // array_of_strings(i-2)
              
              if (s%T(1) >= 0 .and. log_g >= 0 .and. metallicity >= 0) then
                  vals(i) = CalculateSyntheticMagnitude(teff, log_g, metallicity, ierr, wavelengths, fluxes, filter_wavelengths, filter_trans, filter_filepath)
              else
                  vals(i) = -1.0_dp ! Assign an error avlue
                  ierr = 1          ! Indicate an error
              end if
          end do
      else
          ierr = 1 ! Indicate an error if array_of_strings is not allocated
      end if

      if (allocated(array_of_strings)) deallocate(array_of_strings)
  end subroutine data_for_extra_history_columns








  !###########################################################
  !## CUSTOM COLOURS
  !########################################################### 



SUBROUTINE ConvolveSED(wavelengths, fluxes, filter_wavelengths, filter_trans, convolved_flux)
  IMPLICIT NONE
  REAL(DP), DIMENSION(:), INTENT(INOUT) :: wavelengths, fluxes
  REAL(DP), DIMENSION(:), INTENT(INOUT) :: filter_wavelengths, filter_trans
  REAL(DP), DIMENSION(:), ALLOCATABLE :: convolved_flux
  REAL(DP), DIMENSION(:), ALLOCATABLE :: interpolated_filter
  INTEGER :: n

  n = SIZE(wavelengths)

  ! Allocate arrays
  ALLOCATE(interpolated_filter(n))
  ALLOCATE(convolved_flux(n))
  !print *, wavelengths, 'aaaaaa'
  ! Interpolate the filter transmission onto the wavelengths array
  CALL InterpolateArray(filter_wavelengths, filter_trans, wavelengths, interpolated_filter)

  ! Perform convolution (element-wise multiplication)
  convolved_flux = fluxes * interpolated_filter

  ! Deallocate arrays (optional, depending on context)
  DEALLOCATE(interpolated_filter)
END SUBROUTINE ConvolveSED



REAL(DP) FUNCTION CalculateSyntheticMagnitude(temperature, gravity, metallicity, ierr, wavelengths, fluxes, filter_wavelengths, filter_trans, filter_filepath)
    IMPLICIT NONE
    REAL(DP), INTENT(IN) :: temperature, gravity, metallicity
    CHARACTER(LEN=*), INTENT(IN) :: filter_filepath     
    INTEGER, INTENT(OUT) :: ierr
    REAL(DP), DIMENSION(:), ALLOCATABLE, INTENT(INOUT) :: wavelengths, fluxes, filter_wavelengths, filter_trans
    REAL(DP), DIMENSION(:), ALLOCATABLE :: convolved_flux
    REAL(DP), DIMENSION(:), ALLOCATABLE :: interpolated_filter
    REAL(DP) :: bolometric_flux
    INTEGER :: n_wavelengths

INTEGER :: max_size
   INTEGER :: i
   ! Define the kinds for each array
   INTEGER, PARAMETER :: kind_wavelengths = KIND(wavelengths(1))
   INTEGER, PARAMETER :: kind_fluxes = KIND(fluxes(1))
   INTEGER, PARAMETER :: kind_convolved_flux = KIND(convolved_flux(1))
   INTEGER, PARAMETER :: kind_interpolated_filter = KIND(interpolated_filter(1))

    print *, "Filter File Path: ", filter_filepath
    ierr = 0

    ! Load filter data
    CALL LoadFilter(filter_filepath, filter_wavelengths, filter_trans)

    ! Check for invalid gravity input
    IF (gravity == 0.0_DP) THEN
        ierr = 1
        CalculateSyntheticMagnitude = 0.0_DP
        RETURN
    END IF
    !print *, '333333333333333', wavelengths, SIZE(wavelengths)
    ! Determine size of wavelengths
    n_wavelengths = SIZE(wavelengths)

    ! Allocate the interpolated filter array
    ALLOCATE(interpolated_filter(n_wavelengths))

    CALL ConvolveSED(wavelengths, fluxes, filter_wavelengths, filter_trans, convolved_flux)



   ! Wavelengths
   print *, "Filter Wavelengths: "
   print *, "  Size: ", SIZE(filter_wavelengths)
   print *, "  Median: ", filter_wavelengths(SIZE(filter_wavelengths) / 2)
   print *, "  Range: ", MINVAL(filter_wavelengths), " to ", MAXVAL(filter_wavelengths)

   ! Wavelengths
   print *, "Wavelengths: "
   print *, "  Size: ", SIZE(wavelengths)
   print *, "  Median: ", wavelengths(SIZE(wavelengths) / 2)
   print *, "  Range: ", MINVAL(wavelengths), " to ", MAXVAL(wavelengths)

   ! Interpolated Filter
   print *, "Interpolated Filter: "
   print *, "  Size: ", SIZE(interpolated_filter)
   print *, "  Median: ", interpolated_filter(SIZE(interpolated_filter) / 2)
   print *, "  Range: ", MINVAL(interpolated_filter), " to ", MAXVAL(interpolated_filter)

   ! Fluxes
   print *, "Fluxes: "
   print *, "  Size: ", SIZE(fluxes)
   print *, "  Median: ", fluxes(SIZE(fluxes) / 2)
   print *, "  Range: ", MINVAL(fluxes), " to ", MAXVAL(fluxes)

   ! Convolved Fluxes
   print *, "Convolved Fluxes: "
   print *, "  Size: ", SIZE(convolved_flux)
   print *, "  Median: ", convolved_flux(SIZE(convolved_flux) / 2)
   print *, "  Range: ", MINVAL(convolved_flux), " to ", MAXVAL(convolved_flux)
   
      
   ! Open the CSV file for writing
   OPEN(UNIT=10, FILE="full_arrays.csv", STATUS="REPLACE", ACTION="WRITE")

   ! Write a header row to indicate columns
   WRITE(10, '(A)') "Index,Wavelengths,Fluxes,Convolved Fluxes,Interpolated Filter"

   ! Determine the maximum size among the arrays
   max_size = MAX(SIZE(wavelengths), SIZE(fluxes), SIZE(convolved_flux), SIZE(interpolated_filter))

   ! Loop through the arrays and write their values
   DO i = 1, max_size
       WRITE(10, '(I6,",",4(F15.6,","))') i, &
           MERGE(wavelengths(i), 0.0_kind_wavelengths, i <= SIZE(wavelengths)), &
           MERGE(fluxes(i), 0.0_kind_fluxes, i <= SIZE(fluxes)), &
           MERGE(convolved_flux(i), 0.0_kind_convolved_flux, i <= SIZE(convolved_flux)), &
           MERGE(interpolated_filter(i), 0.0_kind_interpolated_filter, i <= SIZE(interpolated_filter))
   END DO

   ! Close the CSV file
   CLOSE(10)

    ! Stop execution
   !STOP
END FUNCTION CalculateSyntheticMagnitude

  SUBROUTINE CalculateBolometricMagnitude(teff, log_g, metallicity, bolometric_magnitude, bolometric_flux, wavelengths, fluxes, sed_filepath)
    IMPLICIT NONE
    REAL(8), INTENT(IN) :: teff, log_g, metallicity
    CHARACTER(LEN=*), INTENT(IN) :: sed_filepath
    REAL, INTENT(OUT) :: bolometric_magnitude, bolometric_flux

    ! Remove INTENT for ALLOCATABLE arrays
    REAL, ALLOCATABLE :: lu_logg(:), lu_meta(:), lu_teff(:)
    CHARACTER(LEN=100), ALLOCATABLE :: file_names(:)

    REAL, DIMENSION(:,:), ALLOCATABLE :: lookup_table
    REAL(DP), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: wavelengths, fluxes
    CHARACTER(LEN=256) :: lookup_file

     lookup_file = TRIM(sed_filepath) // '/lookup_table.csv'

    ! Call to load the lookup table
    CALL LoadLookupTable(lookup_file, lookup_table, file_names, lu_logg, lu_meta, lu_teff)
    ! Interpolate Spectral Energy Distribution
    CALL InterpolateSED(teff, log_g, metallicity, file_names, lu_teff, lu_logg, lu_meta, sed_filepath, wavelengths, fluxes)
    ! Calculate bolometric flux and magnitude
    CALL CalculateBolometricFlux(wavelengths, fluxes, bolometric_magnitude, bolometric_flux)

  END SUBROUTINE CalculateBolometricMagnitude





  !****************************
  !## LOAD LOOKUP TABLE FOR CLOSEST MATCH FOR MODEL
  !****************************


  SUBROUTINE LoadLookupTable(lookup_file, lookup_table, out_file_names, out_logg, out_meta, out_teff)
     IMPLICIT NONE
     CHARACTER(LEN=*), INTENT(IN) :: lookup_file
     REAL, DIMENSION(:,:), ALLOCATABLE, INTENT(OUT) :: lookup_table
     CHARACTER(LEN=100), ALLOCATABLE, INTENT(INOUT) :: out_file_names(:)
     REAL, ALLOCATABLE, INTENT(INOUT) :: out_logg(:), out_meta(:), out_teff(:)

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
     meta_col = GetColumnIndex(headers, "meta")
     teff_col = GetColumnIndex(headers, "teff")

     ! Count the number of rows in the file
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

        IF (logg_col > 0 .AND. columns(logg_col) /= "") THEN
           READ(columns(logg_col), *) out_logg(i)
        ELSE
           out_logg(i) = -1.0
        END IF

        IF (meta_col > 0 .AND. columns(meta_col) /= "") THEN
           READ(columns(meta_col), *) out_meta(i)
        ELSE
           out_meta(i) = -1.0
        END IF

        IF (teff_col > 0 .AND. columns(teff_col) /= "") THEN
           READ(columns(teff_col), *) out_teff(i)
        ELSE
           out_teff(i) = -1.0
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






  !****************************
  !## GENERATE SED FROM LOOKUP TABLE RESULTS
  !****************************

  SUBROUTINE InterpolateSED(teff, log_g, metallicity, file_names, lu_teff, lu_logg, lu_meta, stellar_model_dir, wavelengths, fluxes)
    IMPLICIT NONE
    REAL(8), INTENT(IN) :: teff, log_g, metallicity
    REAL, INTENT(IN) :: lu_teff(:), lu_logg(:), lu_meta(:)
    CHARACTER(LEN=*), INTENT(IN) :: stellar_model_dir
    CHARACTER(LEN=100), INTENT(IN) :: file_names(:)  ! File names of stellar models
    REAL(DP), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: wavelengths, fluxes

    INTEGER, DIMENSION(2) :: closest_indices
    REAL(DP), DIMENSION(:), ALLOCATABLE :: sed1_wavelengths, sed1_flux, sed2_wavelengths, sed2_flux
    REAL(DP), DIMENSION(:), ALLOCATABLE :: sed2_flux_rescaled
    REAL :: weight1, weight2
    CHARACTER(LEN=256) :: sed_dir1, sed_dir2  ! Paths to SED files

    ! Get indices of the two closest stellar models
    CALL GetClosestStellarModels(teff, log_g, metallicity, lu_teff, lu_logg, lu_meta, closest_indices)

    ! Load the first SED
    sed_dir1 = TRIM(stellar_model_dir) // TRIM(file_names(closest_indices(1)))
    CALL LoadSED(sed_dir1, closest_indices(1), sed1_wavelengths, sed1_flux)

    ! Load the second SED
    sed_dir2 = TRIM(stellar_model_dir) // TRIM(file_names(closest_indices(2)))
    CALL LoadSED(sed_dir2, closest_indices(2), sed2_wavelengths, sed2_flux)

    ! Check for a perfect match with the first SED
    IF (lu_teff(closest_indices(1)) == teff .AND. lu_logg(closest_indices(1)) == log_g .AND. &
        lu_meta(closest_indices(1)) == metallicity) THEN
      wavelengths = sed1_wavelengths
      fluxes = sed1_flux
      RETURN
    END IF

    ! Interpolate sed2_flux onto sed1_wavelengths
    ALLOCATE(sed2_flux_rescaled(SIZE(sed1_wavelengths)))
    CALL InterpolateArray(sed2_wavelengths, sed2_flux, sed1_wavelengths, sed2_flux_rescaled)

    ! Calculate interpolation weights based on temperature
    IF (lu_teff(closest_indices(1)) == lu_teff(closest_indices(2))) THEN
      weight1 = 0.5
      weight2 = 0.5
    ELSE
      weight1 = ABS(lu_teff(closest_indices(2)) - teff) / &
                ABS(lu_teff(closest_indices(2)) - lu_teff(closest_indices(1)))
      weight2 = 1.0 - weight1
    END IF

    ! Interpolate fluxes
    ALLOCATE(wavelengths(SIZE(sed1_wavelengths)))
    ALLOCATE(fluxes(SIZE(sed1_flux)))
    wavelengths = sed1_wavelengths
    fluxes = weight1 * sed1_flux + weight2 * sed2_flux_rescaled

    ! Deallocate temporary arrays
    DEALLOCATE(sed2_flux_rescaled)
  END SUBROUTINE InterpolateSED


  SUBROUTINE GetClosestStellarModels(teff, log_g, metallicity, lu_teff, lu_logg, lu_meta, closest_indices)
    IMPLICIT NONE
    REAL(8), INTENT(IN) :: teff, log_g, metallicity
    REAL, INTENT(IN) :: lu_teff(:), lu_logg(:), lu_meta(:)
    INTEGER, DIMENSION(2), INTENT(OUT) :: closest_indices

    INTEGER :: i, n
    REAL :: distance, min_distance1, min_distance2
    INTEGER :: index1, index2

    n = SIZE(lu_teff)
    min_distance1 = HUGE(1.0)
    min_distance2 = HUGE(1.0)

    ! Loop through all stellar models to find the two closest matches
    DO i = 1, n
      distance = SQRT((lu_teff(i) - teff)**2 + (lu_logg(i) - log_g)**2 + (lu_meta(i) - metallicity)**2)

      ! Update the closest and second closest models
      IF (distance < min_distance1) THEN
        min_distance2 = min_distance1
        index2 = index1
        min_distance1 = distance
        index1 = i
      ELSE IF (distance < min_distance2) THEN
        min_distance2 = distance
        index2 = i
      END IF
    END DO

    ! Assign the closest indices to the output array
    closest_indices(1) = index1
    closest_indices(2) = index2
  END SUBROUTINE GetClosestStellarModels


  SUBROUTINE LoadSED(directory, index, wavelengths, flux)
    IMPLICIT NONE
    CHARACTER(LEN=*), INTENT(IN) :: directory
    INTEGER, INTENT(IN) :: index
    REAL(DP), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: wavelengths, flux

    CHARACTER(LEN=512) :: line
    INTEGER :: unit, n_rows, status, i
    REAL :: temp_wavelength, temp_flux

    ! Generate the sed_filepath
    !PRINT *, "Debug: Attempting to open file: ", TRIM(directory)  ! Debugging line

    ! Open the file
    unit = 20
    OPEN(unit, FILE=TRIM(directory), STATUS='OLD', ACTION='READ', IOSTAT=status)
    IF (status /= 0) THEN
      PRINT *, "Error: Could not open file ", TRIM(directory)  ! Print the problematic path
      STOP
    END IF

    ! Skip header lines
    DO
      READ(unit, '(A)', IOSTAT=status) line
      IF (status /= 0) THEN
        PRINT *, "Error: Could not read the file", TRIM(directory)
        STOP
      END IF
      IF (line(1:1) /= "#") EXIT  ! Stop skipping once non-comment is found
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
      wavelengths(i) = temp_wavelength
      flux(i) = temp_flux
    END DO

    CLOSE(unit)
  END SUBROUTINE LoadSED
  
  

  SUBROUTINE LoadFilter(directory, filter_wavelengths, filter_trans)
    IMPLICIT NONE
    CHARACTER(LEN=*), INTENT(IN) :: directory
    REAL(DP), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: filter_wavelengths, filter_trans

    CHARACTER(LEN=512) :: line
    INTEGER :: unit, n_rows, status, i
    REAL :: temp_wavelength, temp_trans

    ! Generate the sed_filepath
    !PRINT *, "Debug: Attempting to open file: ", TRIM(directory)  ! Debugging line

    ! Open the file
    unit = 20
    OPEN(unit, FILE=TRIM(directory), STATUS='OLD', ACTION='READ', IOSTAT=status)
    IF (status /= 0) THEN
      PRINT *, "Error: Could not open file ", TRIM(directory)  ! Print the problematic path
      STOP
    END IF

    ! Skip header line (filter file has just one line header)
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
  !## INTEGRATE OVER SED AND CALCULATE FLUX AND THEN US MAG OFFSET TO GENERATE BOLOMETRIC MAGNITUDE
  !****************************

!  SUBROUTINE CalculateBolometricFlux(wavelengths, fluxes, bolometric_magnitude, bolometric_flux)
!    IMPLICIT NONE
!    REAL, DIMENSION(:), INTENT(IN) :: wavelengths, fluxes
!    REAL, INTENT(OUT) :: bolometric_magnitude, bolometric_flux
!
!    ! Perform trapezoidal integration over wavelengths
!    CALL TrapezoidalIntegration(wavelengths, fluxes, bolometric_flux)
!    PRINT *, wavelengths, fluxes, bolometric_flux
!    IF (bolometric_flux > 0.0) THEN
!       ! Use AB magnitude offset (consistent with Python's -48.6)
!       bolometric_magnitude = -2.5 * LOG10(bolometric_flux) - 48.6
!    ELSE
!       bolometric_magnitude = 99.0  ! Assign a dummy value for invalid flux
!       PRINT *, "Warning: Bolometric flux is zero or negative."
!    END IF
!    PRINT *, bolometric_flux, bolometric_magnitude
!    STOP
!  END SUBROUTINE CalculateBolometricFlux


SUBROUTINE CalculateBolometricFlux(wavelengths, fluxes, bolometric_magnitude, bolometric_flux)
    IMPLICIT NONE
    REAL(DP), DIMENSION(:), INTENT(IN) :: wavelengths, fluxes
    REAL, INTENT(OUT) :: bolometric_magnitude, bolometric_flux
    INTEGER :: i

    ! Validate inputs
    DO i = 1, SIZE(wavelengths) - 1
        IF (wavelengths(i) <= 0.0 .OR. fluxes(i) < 0.0) THEN
            PRINT *, "Invalid input at index", i, ":", wavelengths(i), fluxes(i)
            STOP
        END IF
    END DO
    
    ! Perform trapezoidal integration
    CALL TrapezoidalIntegration(wavelengths, fluxes, bolometric_flux)
    PRINT *, "Wavelengths :", wavelengths
    PRINT *, "Fluxes (erg/cm2/s/A):", fluxes
    PRINT *, "Integrated Bolometric Flux (erg/cm2/s):", bolometric_flux

    ! Validate integration result
    IF (bolometric_flux <= 0.0) THEN
        PRINT *, "Error: Flux integration resulted in non-positive value."
        bolometric_magnitude = 99.0
        RETURN
    END IF

    ! Calculate bolometric magnitude
    bolometric_magnitude = -2.5 * LOG10(bolometric_flux) + 21.1

    PRINT *, "Bolometric magnitude:", bolometric_magnitude
END SUBROUTINE CalculateBolometricFlux


  !****************************
  !## MATHS FUNCTIONS
  !****************************


SUBROUTINE TrapezoidalIntegration(x, y, result)
    IMPLICIT NONE
    REAL(DP), DIMENSION(:), INTENT(IN) :: x, y
    REAL, INTENT(OUT) :: result

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

    ! Validate monotonicity of x
    DO i = 1, n - 1
        IF (x(i+1) <= x(i)) THEN
            PRINT *, "Error: x array is not strictly increasing at index", i
            STOP
        END IF
    END DO

    ! Validate y values
    DO i = 1, n
        IF (y(i) /= y(i)) THEN
            PRINT *, "Error: y array contains NaN at index", i
            STOP
        END IF

        IF (ABS(y(i)) == HUGE(y(i))) THEN
            PRINT *, "Error: y array contains infinity at index", i
            STOP
        END IF
    END DO

    ! Perform trapezoidal integration
    DO i = 1, n - 1
        !PRINT *, "x(i):", x(i), "x(i+1):", x(i+1), "y(i):", y(i), "y(i+1):", y(i+1), "Contribution:", 0.5 * (x(i+1) - x(i)) * (y(i+1) + y(i))
        sum = sum + 0.5 * (x(i+1) - x(i)) * (y(i+1) + y(i))
    END DO

    result = sum
    !PRINT *, "Final integrated result:", result
END SUBROUTINE TrapezoidalIntegration


SUBROUTINE LinearInterpolate(x, y, x_val, y_val)
  IMPLICIT NONE
  REAL(DP), INTENT(IN) :: x(:), y(:), x_val
  REAL(DP), INTENT(OUT) :: y_val
  INTEGER :: i
  REAL(DP) :: slope

  
   PRINT *, "x size: ", SIZE(x), " y size: ", SIZE(y)
   !PRINT *, "x_val: ", x_val
   !PRINT *, "x: ", x
   !PRINT *, "y: ", y

  
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
    PRINT *, "Warning: x_val is below the minimum x."
    y_val = y(1)  ! Assign the first y value
    RETURN
  ELSE IF (x_val > MAXVAL(x)) THEN
    PRINT *, "Warning: x_val is above the maximum x."
    y_val = y(SIZE(y))  ! Assign the last y value
    RETURN
  END IF

  ! Perform interpolation
  DO i = 1, SIZE(x) - 1
    IF (x_val >= x(i) .AND. x_val <= x(i + 1)) THEN
      IF (x(i + 1) == x(i)) THEN
        PRINT *, "Warning: Duplicate x values detected."
        y_val = y(i)  ! Assign current y value
        RETURN
      END IF
      slope = (y(i + 1) - y(i)) / (x(i + 1) - x(i))
      y_val = y(i) + slope * (x_val - x(i))
      RETURN
    END IF
  END DO

  ! Default case (should not occur due to bounds check)
  PRINT *, "Error: Unexpected case in LinearInterpolate."
  y_val = 0.0_DP
END SUBROUTINE LinearInterpolate

SUBROUTINE InterpolateArray(x_in, y_in, x_out, y_out)
  IMPLICIT NONE
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

  ! Perform interpolation for each value in x_out
  DO i = 1, SIZE(x_out)
    CALL LinearInterpolate(x_in, y_in, x_out(i), y_out(i))
  END DO
END SUBROUTINE InterpolateArray


end module run_star_extras



