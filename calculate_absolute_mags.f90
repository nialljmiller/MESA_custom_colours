! Module for constants and utility functions
module utilities
  implicit none
  ! Physical constants (cgs units)
  real(8), parameter :: h = 6.62607015e-27   ! Planck constant (erg·s)
  real(8), parameter :: c = 2.99792458e10    ! Speed of light (cm/s)
  real(8), parameter :: k_B = 1.380649e-16   ! Boltzmann constant (erg/K)
contains

  ! Function to perform trapezoidal integration
  function trapezoidal_integration(x, y, n) result(integral)
    implicit none
    integer, intent(in) :: n
    real(8), intent(in) :: x(n), y(n)
    real(8) :: integral
    integer :: i

    integral = 0.0d0
    do i = 1, n - 1
      integral = integral + 0.5d0 * (x(i+1) - x(i)) * (y(i+1) + y(i))
    end do
  end function trapezoidal_integration

  subroutine get_closest_stellar_models(teff_target, logg_target, teff_array, logg_array, n, closest_indices)
    implicit none
    ! Inputs
    real(8), intent(in) :: teff_target, logg_target   ! Target Teff and log(g)
    real(8), intent(in) :: teff_array(:), logg_array(:)  ! Arrays of Teff and log(g)
    integer, intent(in) :: n                          ! Number of entries in the lookup table
    ! Outputs
    integer, intent(out) :: closest_indices(2)        ! Indices of the two closest models
    ! Local variables
    real(8), allocatable :: distances(:)              ! Array to store distances
    integer :: i, j, temp_index
    real(8) :: temp_distance

    ! Allocate the distances array
    allocate(distances(n))

    ! Calculate Euclidean distances
    do i = 1, n
        distances(i) = sqrt((teff_array(i) - teff_target)**2 + (logg_array(i) - logg_target)**2)
    end do

    ! Initialize closest_indices with dummy values
    closest_indices = (/ -1, -1 /)

    ! Find the two smallest distances
    do i = 1, 2
        temp_distance = huge(1.0d0)  ! Initialize with a large value
        do j = 1, n
            if (distances(j) < temp_distance) then
                temp_distance = distances(j)
                temp_index = j
            end if
        end do
        closest_indices(i) = temp_index
        distances(temp_index) = huge(1.0d0)  ! Exclude this distance from the next search
    end do

    ! Deallocate the distances array
    deallocate(distances)

    print *, "Closest indices found: ", closest_indices
  end subroutine get_closest_stellar_models



  subroutine read_lookup_table(filepath, teff_array, logg_array, file_names, n_models)
    implicit none
    character(len=*), intent(in) :: filepath                ! Path to the lookup table file
    real(8), allocatable, intent(out) :: teff_array(:), logg_array(:)  ! Arrays for Teff and Log(g)
    character(len=256), allocatable, intent(out) :: file_names(:)      ! Array for file names
    integer, intent(out) :: n_models                       ! Number of models in the lookup table
    integer :: unit, iostat, i
    character(len=256) :: line, file_name
    real(8) :: teff, logg
    integer :: count

    ! Open the file
    open(newunit=unit, file=filepath, status='old', action='read', iostat=iostat)
    if (iostat /= 0) then
      print *, "Error: Unable to open file: ", trim(filepath)
      stop
    end if

    ! Count the number of lines (excluding the header)
    count = 0
    do
      read(unit, '(A)', iostat=iostat) line
      if (iostat /= 0) exit
      if (line(1:1) /= '#') count = count + 1
    end do

    n_models = count

    ! Allocate arrays
    allocate(teff_array(n_models), logg_array(n_models), file_names(n_models))

    ! Rewind to start reading the data
    rewind(unit)
    do
      read(unit, '(A)', iostat=iostat) line
      if (iostat /= 0) exit
      if (line(1:1) /= '#') exit  ! Skip the header line
    end do

    ! Read the data into arrays
    i = 1
    do
      read(unit, '(A)', iostat=iostat) line
      if (iostat /= 0) exit
      if (trim(line) == '') cycle  ! Skip empty lines

      ! Parse the line (assumes CSV format: file_name, atmosphere, feh, logg, teff)
      read(line, '(A, 4X, F8.3, 1X, F8.3)', iostat=iostat) file_name, logg, teff
      if (iostat /= 0) then
        print *, "Error: Malformed line in file: ", trim(line)
        stop
      end if

      file_names(i) = trim(file_name)
      teff_array(i) = teff
      logg_array(i) = logg
      i = i + 1
    end do

    ! Close the file
    close(unit)

    print *, "Lookup table loaded successfully with ", n_models, " entries."
  end subroutine read_lookup_table



end module utilities
! Main program


program stellar_analysis
  use utilities
  implicit none

  ! Variables
  character(len=256) :: filename
  character(len=256) :: base_dir, history_file, stellar_model, instrument, vega_sed_file, lookup_table_file
  integer :: iostat, n_models
  real(8), allocatable :: teff_array(:), logg_array(:)
  real(8) :: teff_target, logg_target
  integer, allocatable :: closest_indices(:)

  ! Get the MESA history file path
  print *, "Enter the path to the MESA history file:"
  !read(*, '(A)') filename
  filename = "history.data"

  ! Load paths from 'dir_inlist.txt'
  call load_and_resolve_paths('dir_inlist.txt', base_dir, history_file, stellar_model, instrument, vega_sed_file)

  ! Construct the lookup table file path
  lookup_table_file = trim(stellar_model) // "/lookup_table.csv"

  ! Read the lookup table
  call read_lookup_table(lookup_table_file, teff_array, logg_array, n_models)

  ! Debugging output to confirm lookup table is loaded
  print *, "Lookup table loaded successfully with ", n_models, " entries."

  ! Example target parameters (replace these with real data later)
  teff_target = 5800.0d0
  logg_target = 4.4d0

  ! Allocate array for closest model indices
  allocate(closest_indices(2))

  ! Find the closest stellar models
  call get_closest_stellar_models(teff_target, logg_target, teff_array, logg_array, n_models, closest_indices)

  ! Debugging output for closest indices
  print *, "Closest stellar models indices: ", closest_indices
  print *, "Closest model parameters:"
  print *, "Model 1: Teff=", teff_array(closest_indices(1)), ", Log(g)=", logg_array(closest_indices(1))
  print *, "Model 2: Teff=", teff_array(closest_indices(2)), ", Log(g)=", logg_array(closest_indices(2))

  ! Cleanup
  deallocate(teff_array, logg_array, closest_indices)

end program stellar_analysis


subroutine read_mesa_history(filename)
  use utilities
  implicit none
  character(len=*), intent(in) :: filename
  integer :: iostat, unit, header_line_idx, n_rows, n_cols, i, j
  character(len=256) :: line
  character(len=32), allocatable :: headers(:)
  real(8), allocatable :: data(:,:)

  open(newunit=unit, file=filename, status='old', action='read', iostat=iostat)
  if (iostat /= 0) then
    print *, "Error: Cannot open file ", trim(filename)
    stop
  end if

  ! Locate the header line
  header_line_idx = -1
  i = 0
  do
    read(unit, '(A)', iostat=iostat) line
    if (iostat /= 0) exit
    i = i + 1
    if (index(trim(line), 'model_number') > 0) then
      header_line_idx = i
      exit
    end if
  end do

  if (header_line_idx == -1) then
    print *, "Error: Header line not found in file."
    close(unit)
    stop
  end if

  ! Rewind and read header line
  rewind(unit)
  do i = 1, header_line_idx
    read(unit, '(A)', iostat=iostat) line
  end do

  ! Parse header line into column names
  n_cols = 0
  do i = 1, len_trim(line)
    if (line(i:i) /= ' ') then
      if (i == 1 .or. line(i-1:i-1) == ' ') n_cols = n_cols + 1
    end if
  end do
  allocate(headers(n_cols))
  read(line, *) (headers(i), i = 1, n_cols)

  print *, "Columns found: ", n_cols
  do i = 1, n_cols
    print *, "Column ", i, ": ", trim(headers(i))
  end do

  ! Allocate data array
  rewind(unit)
  n_rows = 0
  do
    read(unit, '(A)', iostat=iostat)
    if (iostat /= 0) exit
    n_rows = n_rows + 1
  end do
  n_rows = n_rows - header_line_idx  ! Subtract header and metadata rows

  allocate(data(n_rows, n_cols))

  ! Read data into array
  rewind(unit)
  do i = 1, header_line_idx
    read(unit, '(A)', iostat=iostat)  ! Skip to data section
  end do

  do i = 1, n_rows
    read(unit, *, iostat=iostat) (data(i, j), j = 1, n_cols)
    if (iostat /= 0) exit
  end do

  print *, "Data successfully read into array with dimensions: ", n_rows, "x", n_cols

  close(unit)
end subroutine read_mesa_history






! Function to calculate bolometric magnitude
function calculate_bolometric_magnitude(wavelength, flux, n) result(bol_mag)
  use utilities
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: wavelength(n), flux(n)
  real(8) :: bol_mag, bolometric_flux

  ! Calculate bolometric flux using trapezoidal integration
  bolometric_flux = trapezoidal_integration(wavelength, flux, n)

  ! Calculate bolometric magnitude
  bol_mag = -2.5d0 * log10(bolometric_flux) - 48.6d0
end function calculate_bolometric_magnitude





subroutine load_and_resolve_paths(filepath, base_dir, history_file, stellar_model, instrument, vega_sed_file)
  implicit none
  character(len=*), intent(in) :: filepath
  character(len=256), intent(out) :: base_dir, history_file, stellar_model, instrument, vega_sed_file
  integer :: unit, iostat, sep_pos
  character(len=256) :: line, key, value

  ! Open the file for reading
  open(newunit=unit, file=filepath, status='old', action='read', form='formatted', iostat=iostat)
  if (iostat /= 0) then
    print *, "Error: Cannot open file ", trim(filepath)
    stop
  end if

  ! Initialize outputs as empty
  base_dir = ''
  history_file = ''
  stellar_model = ''
  instrument = ''
  vega_sed_file = ''

  ! Read the file line by line
  do
    read(unit, '(A)', iostat=iostat) line
    if (iostat /= 0) exit  ! End of file

    ! Skip empty lines or comments
    line = trim(adjustl(line))
    if (len_trim(line) == 0 .or. line(1:1) == '#') cycle

    ! Find the separator '='
    sep_pos = index(line, '=')
    if (sep_pos > 0) then
      key = trim(adjustl(line(1:sep_pos-1)))
      value = trim(adjustl(line(sep_pos+1:)))
      
      ! Assign values based on the key
      select case (trim(key))
      case ('base_dir')
        base_dir = value
      case ('history_file')
        history_file = value
      case ('stellar_model')
        stellar_model = value
      case ('instrument')
        instrument = value
      case ('vega_sed_file')
        vega_sed_file = value
      case default
        print *, "Warning: Unrecognized key in file: ", trim(key)
      end select
    else
      print *, "Warning: Skipping malformed line: ", trim(line)
    end if
  end do

  close(unit)
end subroutine load_and_resolve_paths



