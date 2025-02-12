program game_tree
    implicit none
    integer, parameter :: max_length = 255
    character(len=max_length) :: filename

    ! Tree structure
    type :: TreeNode
        character(len=max_length) :: user_id
        character(len=max_length) :: game_name
        ! NOTE: just store time_played as string since we don't need to do
        !       any operations on it.
        real :: time_played
        type(TreeNode), pointer :: left => null()
        type(TreeNode), pointer :: right => null()
    end type TreeNode

    type(TreeNode), pointer :: root => null()

    ! Open CSV and read data
    call get_command_argument(1, filename)
    if (len_trim(filename) == 0) then
        print *, "Error: No filename provided. Usage: ./game_tree <csv_file>"
        stop
    end if
    call load_csv(filename, root)

contains

    ! Function to insert node into the tree
    recursive subroutine insert_node(root, user_id, game_name, time_played)
        type(TreeNode), pointer :: root
        character(len=max_length), intent(in) :: user_id, game_name
        real, intent(in) :: time_played
        character(len=max_length) :: output

        if (.not. associated(root)) then
            allocate(root)
            root%user_id = user_id
            root%game_name = game_name
            root%time_played = time_played
            root%left => null()
            root%right => null()
        else if ((root%user_id == user_id) .and. (root%game_name == game_name)) then
            ! Allow updating the tree if the previous entry of time_played is 0.0
            ! and the current entry is different
            ! non-zero play times are added together
            if (root%time_played <= 1.0e-6) then
                if (time_played <= 1.0e-6) then
                    print *, "Ignoring duplicate of user_id: ", trim(user_id), " game: ", trim(game_name)
                else
                    root%time_played = time_played
                endif
            else
                if (time_played <= 1.0e-6) then
                    print *, "Ignoring duplicate of user_id: ", trim(user_id), " game: ", trim(game_name)
                else
                    root%time_played = root%time_played + time_played
                endif
            endif
        else if (user_id < root%user_id) then
            call insert_node(root%left, user_id, game_name, time_played)
        else
            call insert_node(root%right, user_id, game_name, time_played)
        endif
    end subroutine insert_node

    ! Read CSV and insert into tree
    subroutine load_csv(filename, root)
        character(len=*), intent(in) :: filename
        type(TreeNode), pointer :: root
        character(len=max_length) :: user_id, game_name, behavior
        real :: time_played
        character(len=max_length) :: line
        integer :: unit, ios
        integer :: i, len_line, start_index

        open(newunit=unit, file=filename, status='old', action='read', iostat=ios)
        if (ios /= 0) then
            print *, "Error opening file."
            stop
        end if

        ! discard first line
        read(unit, '(A)', iostat=ios) line

        do
            read(unit, '(A)', iostat=ios) line
            if (ios /= 0) exit

            ! Parse CSV line while handling quotes
            user_id = ""; game_name = ""; behavior = ""
            time_played = 0.0
            ! Start at second character to skip first quote
            i = -1; len_line = len_trim(line)

            ! Extract user_id
            i = index(line, ",")
            user_id = line(2:i-1)
            ! skip comma and first quote
            i = i + 2

            ! Extract game_name
            start_index = i
            do while (i <= len_line)
                if (line(i:i+1) == '",') then
                    exit
                endif
                i = i + 1
            end do
            game_name = line(start_index:i-1)
            i = i + 2

            ! Extract behavior
            start_index = i
            i = index(line(start_index:), ",")
            behavior = line(start_index:start_index+i-2)
            i = start_index + i

            if (behavior == "purchase") then
                time_played = 0.0
            else
                ! Extract time_played
                start_index = i
                i = index(line(start_index:), ",")
                read(line(start_index:start_index+i-2), *) time_played
            endif

            ! Insert into tree
            call insert_node(root, user_id, game_name, time_played)
        end do

         close(unit)
     end subroutine load_csv

end program game_tree
